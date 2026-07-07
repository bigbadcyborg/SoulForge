"""Tests for runtime chat model switch and import."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from textual.app import App
from textual.widgets import RadioButton

from app.core import model_catalog
from app.core.chat_controller import ChatController
from app.core.config import (
    AgentModelProfileConfig,
    AppConfig,
    ModelConfig,
    load_config,
    save_agents,
    save_chat_model,
)
from app.core.model_catalog import (
    import_chat_model,
    list_available_chat_models,
    resolve_chat_model_selection,
    validate_gguf_source,
)
from app.tui.widgets import ModelSelectionModal


def _write_config(tmp_path: Path, **overrides) -> Path:
    chat = tmp_path / "models" / "chat.gguf"
    embed = tmp_path / "models" / "embed.gguf"
    chat.parent.mkdir(parents=True, exist_ok=True)
    chat.write_bytes(b"chat")
    embed.write_bytes(b"embed")

    data = {
        "model": {
            "chatModelPath": str(chat),
            "embeddingModelPath": str(embed),
            "contextSize": 8192,
            "gpuLayers": 0,
        },
        "features": {"rag": False, "soul": False, "memory": False},
        "rag": {"dbPath": str(tmp_path / "chroma"), "docsPath": str(tmp_path / "docs")},
        "memory": {
            "userFile": str(tmp_path / "memory" / "user.md"),
            "memoryFile": str(tmp_path / "memory" / "memory.md"),
            "sessionFile": str(tmp_path / "memory" / "session.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "skills" / "active"),
            "archivedPath": str(tmp_path / "skills" / "archived"),
            "registryPath": str(tmp_path / "skills" / "registry.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "tasks" / "kanban.json")},
        "sessions": {"storePath": str(tmp_path / "sessions")},
        "logging": {"logPath": str(tmp_path / "test.log"), "level": "info"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in data and isinstance(data[key], dict):
            data[key].update(value)
        else:
            data[key] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _minimal_dirs(tmp_path: Path) -> None:
    for sub in ("skills/active", "skills/archived", "memory", "sessions", "tasks"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks" / "kanban.json").write_text('{"tasks": []}', encoding="utf-8")
    (tmp_path / "skills" / "registry.json").write_text('{"skills": []}', encoding="utf-8")


@pytest.fixture
def model_env(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(model_catalog, "MODELS_DIR", models_dir)
    monkeypatch.setattr(model_catalog, "PROJECT_ROOT", tmp_path)
    _minimal_dirs(tmp_path)
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    return tmp_path, models_dir, config, config_path


def test_list_available_chat_models_excludes_embedding(model_env) -> None:
    tmp_path, models_dir, config, _ = model_env
    (models_dir / "alpha.gguf").write_bytes(b"a")
    (models_dir / "beta.gguf").write_bytes(b"b")
    config.model.embedding_model_path = str(models_dir / "beta.gguf")

    names = [path.name for path in list_available_chat_models(config)]
    assert "alpha.gguf" in names
    assert "beta.gguf" not in names


def test_validate_gguf_source_rejects_non_gguf(model_env) -> None:
    tmp_path, _, _, _ = model_env
    bad = tmp_path / "model.bin"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="Only .gguf"):
        validate_gguf_source(bad)


def test_import_chat_model_copies_file(model_env) -> None:
    tmp_path, models_dir, _, _ = model_env
    source = tmp_path / "external" / "imported.gguf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"gguf-data")

    progress_calls = []
    dest = import_chat_model(source, on_progress=lambda c, t: progress_calls.append((c, t)))
    assert dest.parent == models_dir.resolve()
    assert dest.name == "imported.gguf"
    assert dest.read_bytes() == b"gguf-data"
    assert progress_calls  # Should have at least one call


def test_import_chat_model_collision_suffix(model_env) -> None:
    tmp_path, models_dir, _, _ = model_env
    (models_dir / "dup.gguf").write_bytes(b"old")
    source = tmp_path / "dup.gguf"
    source.write_bytes(b"new")

    dest = import_chat_model(source)
    assert dest.name == "dup-1.gguf"
    assert dest.read_bytes() == b"new"


def test_import_chat_model_rejects_existing_models_dir_file(model_env) -> None:
    _, models_dir, _, _ = model_env
    existing = models_dir / "local.gguf"
    existing.write_bytes(b"x")
    with pytest.raises(ValueError, match="already in ./models/"):
        import_chat_model(existing)


def test_resolve_chat_model_selection_partial_match(model_env) -> None:
    _, models_dir, config, _ = model_env
    (models_dir / "NemoMix.gguf").write_bytes(b"x")
    resolved = resolve_chat_model_selection("Nemo", config)
    assert resolved.name == "NemoMix.gguf"


def test_resolve_chat_model_selection_ambiguous(model_env) -> None:
    _, models_dir, config, _ = model_env
    (models_dir / "alpha-v1.gguf").write_bytes(b"x")
    (models_dir / "alpha-v2.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_chat_model_selection("alpha", config)


def test_save_chat_model_updates_yaml(model_env) -> None:
    _, _, config, config_path = model_env
    config.model.chat_model_path = "./models/new-chat.gguf"
    save_chat_model(config, config_path)
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["model"]["chatModelPath"] == "./models/new-chat.gguf"


def test_save_agents_updates_yaml(model_env) -> None:
    _, _, config, config_path = model_env
    config.agents.model_profiles["critic"] = AgentModelProfileConfig(
        chat_model_path="./models/critic.gguf",
        residency="resident",
    )
    config.agents.roles["critic"].model_profile = "critic"

    save_agents(config, config_path)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["agents"]["modelProfiles"]["critic"]["chatModelPath"] == "./models/critic.gguf"
    assert saved["agents"]["roles"]["critic"]["modelProfile"] == "critic"


def test_switch_chat_model_updates_config_and_runtime(model_env, monkeypatch) -> None:
    _, models_dir, config, config_path = model_env
    (models_dir / "other.gguf").write_bytes(b"x")
    controller = ChatController(config)
    controller.runtime = MagicMock()
    controller.runtime.load_chat_model.return_value = MagicMock()
    monkeypatch.setattr(
        "app.core.chat_controller.save_chat_model",
        lambda cfg, path=None: None,
    )

    name = controller.switch_chat_model("other.gguf")
    assert name == "other.gguf"
    assert controller.config.model.chat_model_path.endswith("other.gguf")
    controller.runtime.unload_chat_model.assert_called_once()
    controller.runtime.load_chat_model.assert_called_once()


def test_switch_chat_model_rolls_back_on_failure(model_env, monkeypatch) -> None:
    _, models_dir, config, _ = model_env
    (models_dir / "bad.gguf").write_bytes(b"x")
    controller = ChatController(config)
    controller.runtime = MagicMock()
    controller.runtime.load_chat_model.side_effect = [RuntimeError("load failed"), MagicMock()]
    old_path = config.model.chat_model_path

    with pytest.raises(RuntimeError, match="load failed"):
        controller.switch_chat_model("bad.gguf", persist=False)

    assert controller.config.model.chat_model_path == old_path
    assert controller.runtime.load_chat_model.call_count == 2


def test_set_agent_role_model_isolates_shared_profile(model_env, monkeypatch) -> None:
    _, models_dir, config, _ = model_env
    (models_dir / "critic.gguf").write_bytes(b"x")
    controller = ChatController(config)
    controller.runtime = MagicMock()
    monkeypatch.setattr(
        "app.core.chat_controller.save_agents",
        lambda cfg, path=None: None,
    )

    message = controller.set_agent_role_model("critic", "critic.gguf")

    assert "Role 'critic'" in message
    assert config.agents.roles["critic"].model_profile == "critic"
    assert config.agents.roles["executor"].model_profile == "critic_executor"
    assert config.agents.model_profiles["critic"].chat_model_path.endswith("critic.gguf")
    assert config.agents.model_profiles["critic"].residency == "resident"
    controller.runtime.unload_chat_profile.assert_called_once_with("critic")


def test_set_agent_role_model_does_not_touch_shared_creator_profile(
    model_env, monkeypatch
) -> None:
    _, models_dir, config, _ = model_env
    (models_dir / "coder.gguf").write_bytes(b"x")
    controller = ChatController(config)
    controller.runtime = MagicMock()
    monkeypatch.setattr(
        "app.core.chat_controller.save_agents",
        lambda cfg, path=None: None,
    )

    # 'creator' names both a role and the shared profile used by synthesizer;
    # rerouting the role must not mutate the shared profile.
    message = controller.set_agent_role_model("creator", "coder.gguf")

    assert "Role 'creator'" in message
    assert config.agents.roles["creator"].model_profile == "creator_role"
    assert config.agents.roles["synthesizer"].model_profile == "creator"
    assert config.agents.model_profiles["creator"].chat_model_path is None
    assert config.agents.model_profiles["creator_role"].chat_model_path.endswith(
        "coder.gguf"
    )
    controller.runtime.unload_chat_profile.assert_called_once_with("creator_role")


def test_set_agent_role_model_can_inherit_chat_model(model_env, monkeypatch) -> None:
    _, models_dir, config, _ = model_env
    (models_dir / "critic.gguf").write_bytes(b"x")
    config.agents.model_profiles["critic"] = AgentModelProfileConfig(
        chat_model_path="models/critic.gguf",
        residency="resident",
    )
    config.agents.roles["critic"].model_profile = "critic"
    controller = ChatController(config)
    controller.runtime = MagicMock()
    monkeypatch.setattr(
        "app.core.chat_controller.save_agents",
        lambda cfg, path=None: None,
    )

    message = controller.set_agent_role_model("critic", "inherit")

    assert "inherits /model" in message
    assert config.agents.roles["critic"].model_profile == "critic"
    assert config.agents.model_profiles["critic"].chat_model_path is None
    controller.runtime.unload_chat_profile.assert_called_once_with("critic")


def test_import_chat_model_with_switch_after(model_env, monkeypatch) -> None:
    tmp_path, models_dir, config, _ = model_env
    source = tmp_path / "fresh.gguf"
    source.write_bytes(b"data")
    controller = ChatController(config)
    controller.runtime = MagicMock()
    controller.runtime.load_chat_model.return_value = MagicMock()
    monkeypatch.setattr(
        "app.core.chat_controller.save_chat_model",
        lambda cfg, path=None: None,
    )

    message = controller.import_chat_model(source, switch_after=True, persist=False)
    assert "Imported model as fresh.gguf" in message
    assert "Switched to fresh.gguf" in message
    assert (models_dir / "fresh.gguf").exists()
    controller.runtime.unload_chat_model.assert_called_once()


def test_model_selection_modal_returns_model_name_for_selection() -> None:
    class ModelSelectionHarness(App):
        def __init__(self, select_index: int | None = None) -> None:
            super().__init__()
            self.select_index = select_index
            self.result = None

        def on_mount(self) -> None:
            modal = ModelSelectionModal(["alpha.gguf", "beta.gguf"], "alpha.gguf")
            self.push_screen(modal, self._store_result)

        def _store_result(self, result) -> None:
            self.result = result
            self.exit()

    async def run_modal(select_index: int | None = None) -> str | None:
        app = ModelSelectionHarness(select_index)
        async with app.run_test() as pilot:
            await pilot.pause()
            if select_index is not None:
                modal = app.screen_stack[-1]
                buttons = list(modal.query(RadioButton))
                buttons[select_index].value = True
                await pilot.pause()
            await pilot.click("#switch-model-button")
            await pilot.pause()
        return app.result

    assert asyncio.run(run_modal()) == "alpha.gguf"
    assert asyncio.run(run_modal(1)) == "beta.gguf"
