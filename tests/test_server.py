"""Tests for the local API server and command router."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.command_router import CommandResult, CommandRouter
from app.server.api import create_app


class FakeFeatures:
    def __init__(self, enabled: dict[str, bool]) -> None:
        self._enabled = enabled

    def is_enabled(self, key: str) -> bool:
        return self._enabled.get(key, False)

    def as_dict(self) -> dict[str, bool]:
        return dict(self._enabled)

    def format_list(self) -> str:
        return "\n".join(f"{k}: {'on' if v else 'off'}" for k, v in self._enabled.items())

    def summary(self) -> str:
        return ", ".join(k for k, v in self._enabled.items() if v) or "none"

    def set_enabled(self, key: str, value: bool) -> None:
        if key not in self._enabled:
            raise KeyError(f"Unknown feature '{key}'")
        self._enabled[key] = value


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []
        self.warmed = False
        self.vision_loaded = False

    def create_vision_completion(self, image_bytes: bytes, prompt: str) -> str:
        self.calls.append((image_bytes, prompt))
        return f"vision saw {len(image_bytes)} bytes: {prompt or 'describe'}"

    def warm_resident_profiles(self):
        self.warmed = True
        return []

    def preload_vision_model(self) -> None:
        self.vision_loaded = True

    def set_load_listener(self, listener) -> None:
        self._listener = listener


class FakeController:
    """Minimal stand-in exercising the paths the server/router use."""

    def __init__(
        self, *, auth_token: str = "", streaming: bool = True, vision: bool = False
    ) -> None:
        self.config = SimpleNamespace(
            server=SimpleNamespace(auth_token=auth_token),
            tools=SimpleNamespace(shell_allowlist=["git status"]),
            vision=SimpleNamespace(enabled=vision),
            transcription=SimpleNamespace(
                model_size="small", device="cpu", compute_type="int8", language=""
            ),
        )
        self.model_name = "test.gguf"
        self.loaded = True
        self.compute_backend = "CPU"
        self.turn_count = 3
        self.features = FakeFeatures(
            {"streaming": streaming, "show_sources": False, "rag": True}
        )
        self.runtime = FakeRuntime()
        self.messages: list[dict] = []
        self.vision_model = ""
        self.last_retrieved_chunks: list = []
        self.skill_manager = SimpleNamespace(
            list_skills=lambda status="active": (
                [{"name": "demo", "description": "a demo skill"}]
                if status == "active"
                else []
            )
        )
        self._pending_raw_reply = ""
        self.finalized: list[str] = []
        self.completed = 0

    # chat flow
    def add_user_turn(self, text: str):
        self._last = text
        return []

    def stream_reply(self):
        for token in ["Hel", "lo ", "world"]:
            yield token
        self._pending_raw_reply = "Hello world"

    def full_reply(self) -> str:
        return "Hello world"

    def finalize_assistant_reply(self, raw: str):
        self.finalized.append(raw)
        return SimpleNamespace(display_text=raw, has_pending=False, pending=[])

    def complete_turn(self):
        self.completed += 1
        return SimpleNamespace(message="", has_suggestion=False)

    # command-router targets
    def run_health_check(self) -> str:
        return "All systems go"

    def features_summary(self) -> str:
        return self.features.summary()

    def set_feature(self, key: str, enabled: bool) -> None:
        self.features.set_enabled(key, enabled)

    def reload_soul(self) -> None:  # returns None, like the real controller
        self.reloaded_soul = True

    # session load
    def load(self) -> None:
        self.loaded = True
        self.load_called = True

    def switch_chat_model(self, name, *, persist=True) -> str:
        self.model_name = name
        self.loaded = True
        return name

    # rag
    def enable_rag(self, sources=None) -> None:
        self.features.set_enabled("rag", True)

    def disable_rag(self) -> None:
        self.features.set_enabled("rag", False)

    def get_rag_status(self) -> dict:
        return {
            "enabled": True,
            "selected_sources": None,
            "available_sources": ["a.md", "b.md"],
        }

    def get_rag_stats(self) -> dict:
        return {"sources": ["a.md", "b.md"], "chunk_count": 42}

    # vision
    def models_info(self) -> dict:
        return {
            "chat_model": self.model_name,
            "available": ["a.gguf", "b.gguf"],
            "roles": {"creator": "a.gguf", "critic": "(inherit)"},
            "vision": {"enabled": bool(self.vision_model), "model": "", "mmproj": ""},
        }

    def format_vision_view(self) -> str:
        return f"Vision model: {self.vision_model or '(disabled)'}"

    def set_vision_model(self, model, mmproj=None, *, handler=None, persist=True) -> str:
        self.vision_model = model
        return f"Vision model set to {model}."

    def disable_vision(self, *, persist=True) -> str:
        self.vision_model = ""
        return "Vision model disabled."

    # tasks / sources
    def get_board_view(self) -> str:
        return "Backlog: (empty)"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FakeController()))


def test_ping(client: TestClient) -> None:
    r = client.get("/api/ping")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "test.gguf"
    assert body["ready"] is True


def test_commands_listing(client: TestClient) -> None:
    r = client.get("/api/commands")
    assert r.status_code == 200
    names = r.json()["commands"]
    assert "health" in names
    assert "agents" in names


def test_command_health(client: TestClient) -> None:
    r = client.post("/api/command", json={"name": "health"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "message"
    assert "systems go" in body["text"]


def test_command_unknown(client: TestClient) -> None:
    r = client.post("/api/command", json={"name": "nope"})
    assert r.json()["kind"] == "error"
    assert r.json()["success"] is False


def test_auth_enforced_when_token_set() -> None:
    app = create_app(FakeController(auth_token="secret"))
    client = TestClient(app)
    # ping is unauthenticated (health probe)
    assert client.get("/api/ping").status_code == 200
    # command requires the token
    assert client.post("/api/command", json={"name": "health"}).status_code == 401
    ok = client.post(
        "/api/command",
        json={"name": "health"},
        headers={"X-SoulForge-Token": "secret"},
    )
    assert ok.status_code == 200


def test_ws_chat_streams_tokens(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi"})
        frames = []
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame["type"] == "done":
                break
    types = [f["type"] for f in frames]
    assert types.count("token") == 3
    assert "final" in types
    tokens = "".join(f["token"] for f in frames if f["type"] == "token")
    assert tokens == "Hello world"


def test_ws_chat_rejects_empty(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "   "})
        frame = ws.receive_json()
        assert frame["type"] == "error"


# -- session start / deferred load --------------------------------------


def test_run_session_load_loads_all() -> None:
    from app.server.api import run_session_load

    controller = FakeController(vision=True)
    controller.loaded = False
    state = {"stage": "starting", "vision_loaded": False, "loading": True}
    request = SimpleNamespace(chat_model=None, load_agents=True, load_vision=True)
    run_session_load(controller, request, state)
    assert controller.loaded is True
    assert controller.runtime.warmed is True
    assert state["vision_loaded"] is True
    assert state["stage"] == "ready"
    assert state["loading"] is False


def test_run_session_load_switches_model() -> None:
    from app.server.api import run_session_load

    controller = FakeController()
    state = {"stage": "starting", "vision_loaded": False, "loading": True}
    request = SimpleNamespace(
        chat_model="other.gguf", load_agents=False, load_vision=False
    )
    run_session_load(controller, request, state)
    assert controller.model_name == "other.gguf"
    assert controller.runtime.warmed is False


def test_session_start_endpoint(client: TestClient) -> None:
    r = client.post(
        "/api/session/start",
        json={"chat_model": None, "load_agents": False, "load_vision": False},
    )
    assert r.status_code == 200
    assert r.json()["started"] is True


def test_ping_exposes_stage_and_vision(client: TestClient) -> None:
    body = client.get("/api/ping").json()
    assert "stage" in body
    assert "vision_loaded" in body


# -- snapshot -----------------------------------------------------------


def test_snapshot_disabled_returns_400(client: TestClient) -> None:
    r = client.post(
        "/api/snapshot",
        files={"image": ("s.png", b"pngdata", "image/png")},
        data={"prompt": "what is this"},
    )
    assert r.status_code == 400


def test_snapshot_runs_vision_and_injects() -> None:
    controller = FakeController(vision=True)
    app = create_app(controller)
    client = TestClient(app)
    r = client.post(
        "/api/snapshot",
        files={"image": ("s.png", b"0123456789", "image/png")},
        data={"prompt": "read it", "inject": "true"},
    )
    assert r.status_code == 200
    assert "vision saw 10 bytes: read it" in r.json()["text"]
    # A user+assistant pair is injected for follow-up context.
    assert len(controller.messages) == 2
    assert controller.messages[0]["role"] == "user"
    assert controller.messages[1]["role"] == "assistant"


# -- transcribe ---------------------------------------------------------


class FakeTranscriber:
    def transcribe_wav(self, wav_bytes: bytes, language: str = "") -> str:
        return f"transcribed {len(wav_bytes)} bytes ({language or 'auto'})"


def test_transcribe_endpoint() -> None:
    app = create_app(FakeController(), transcriber=FakeTranscriber())
    client = TestClient(app)
    r = client.post(
        "/api/transcribe",
        files={"audio": ("clip.wav", b"RIFFxxxx", "audio/wav")},
        data={"language": "en"},
    )
    assert r.status_code == 200
    assert "transcribed 8 bytes (en)" in r.json()["text"]


# -- command router unit tests -------------------------------------------


def test_router_dispatch_strips_slash() -> None:
    router = CommandRouter(FakeController())
    result = router.dispatch("/health")
    assert result.kind == "message"
    assert "systems go" in result.text


def test_router_features_toggle() -> None:
    controller = FakeController()
    router = CommandRouter(controller)
    result = router.dispatch("features", "streaming off")
    assert result.success is True
    assert controller.features.is_enabled("streaming") is False


def test_router_features_unknown_key() -> None:
    router = CommandRouter(FakeController())
    result = router.dispatch("features", "bogus on")
    assert result.success is False


def test_router_status_structured() -> None:
    result = CommandRouter(FakeController()).dispatch("status")
    assert result.kind == "data"
    assert result.data["model"] == "test.gguf"
    assert result.data["turn_count"] == 3


def test_command_result_helpers() -> None:
    assert CommandResult.message("x").kind == "message"
    assert CommandResult.error("x").success is False
    assert CommandResult.structured("x", {"a": 1}).data == {"a": 1}


# -- expanded command coverage ------------------------------------------


def test_router_rag_status_is_readable() -> None:
    # Regression: /rag with no args used to return only the label "RAG status".
    result = CommandRouter(FakeController()).dispatch("rag")
    assert result.kind == "data"
    assert "RAG: on" in result.text
    assert "Indexed chunks: 42" in result.text
    assert "a.md" in result.text
    assert "RAG status." != result.text


def test_router_rag_on_off() -> None:
    controller = FakeController()
    router = CommandRouter(controller)
    assert router.dispatch("rag", "off").success is True
    assert controller.features.is_enabled("rag") is False


def test_router_tasks_and_skills() -> None:
    router = CommandRouter(FakeController())
    assert "Backlog" in router.dispatch("tasks").text
    assert "demo" in router.dispatch("skills").text


def test_router_models_info() -> None:
    result = CommandRouter(FakeController()).dispatch("models", "info")
    assert result.kind == "data"
    assert "available" in result.data
    assert "roles" in result.data
    assert "vision" in result.data


def test_router_models_vision() -> None:
    controller = FakeController()
    router = CommandRouter(controller)
    # set model + mmproj
    result = router.dispatch("models", "vision qwen.gguf mmproj.gguf")
    assert result.success is True
    assert controller.vision_model == "qwen.gguf"
    # view
    assert "qwen.gguf" in router.dispatch("models", "vision").text
    # disable
    router.dispatch("models", "vision off")
    assert controller.vision_model == ""


def test_reload_soul_returns_string_not_none() -> None:
    # Regression: reload_soul() returns None; the handler used to forward that
    # as CommandResult text, which 500'd the API (Pydantic rejects None text).
    controller = FakeController()
    result = CommandRouter(controller).dispatch("reload-soul")
    assert result.kind == "message"
    assert isinstance(result.text, str) and result.text
    assert controller.reloaded_soul is True


def test_reload_soul_over_http_is_200() -> None:
    app = create_app(FakeController())
    client = TestClient(app)
    r = client.post("/api/command", json={"name": "reload-soul"})
    assert r.status_code == 200
    assert r.json()["text"]


def test_command_result_coerces_none_text() -> None:
    assert CommandResult(kind="message", text=None).to_dict()["text"] == ""


def test_router_sources_empty() -> None:
    result = CommandRouter(FakeController()).dispatch("sources")
    assert result.kind == "message"
    assert result.text  # format_sources_detail returns a friendly no-sources note


def test_router_covers_full_command_set() -> None:
    names = set(CommandRouter(FakeController()).command_names())
    # A representative sweep across every command group.
    for expected in [
        "help", "status", "rag", "ingest", "sources", "memory-review",
        "memory-forget", "skills", "crystallize", "curator", "tasks",
        "task-new", "agents", "session-save", "tools", "tool-approve",
    ]:
        assert expected in names


def test_memory_forget_clears_episodic() -> None:
    controller = FakeController()
    controller.episodic_cleared = 0

    def clear_episodic():
        controller.episodic_cleared += 1
        return "Cleared 3 remembered conversation turn(s)."

    controller.clear_episodic_memory = clear_episodic
    result = CommandRouter(controller).dispatch("memory-forget")
    assert result.success is True
    assert "Cleared 3" in result.text
    assert controller.episodic_cleared == 1


# -- config -------------------------------------------------------------


def test_server_config_defaults_and_load(tmp_path) -> None:
    import yaml

    from app.core.config import ServerConfig, load_config

    assert ServerConfig().host == "127.0.0.1"
    assert ServerConfig().port == 8765

    data = {
        "model": {"chatModelPath": "x.gguf", "embeddingModelPath": "e.gguf"},
        "server": {"host": "0.0.0.0", "port": 9000, "authToken": "abc"},
        "rag": {"dbPath": str(tmp_path / "c"), "docsPath": str(tmp_path / "d")},
        "memory": {
            "userFile": str(tmp_path / "u.md"),
            "memoryFile": str(tmp_path / "m.md"),
            "sessionFile": str(tmp_path / "s.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "sa"),
            "archivedPath": str(tmp_path / "sr"),
            "registryPath": str(tmp_path / "r.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "k.json")},
        "sessions": {"storePath": str(tmp_path / "sess")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_config(path)
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9000
    assert config.server.auth_token == "abc"


def test_vision_config_load(tmp_path) -> None:
    import yaml

    from app.core.config import VisionConfig, load_config

    assert VisionConfig().enabled is False

    data = {
        "model": {"chatModelPath": "x.gguf", "embeddingModelPath": "e.gguf"},
        "vision": {
            "modelPath": "./models/vision/llava.gguf",
            "mmprojPath": "./models/vision/mmproj.gguf",
            "chatHandler": "llava-1-6",
            "maxTokens": 256,
        },
        "rag": {"dbPath": str(tmp_path / "c"), "docsPath": str(tmp_path / "d")},
        "memory": {
            "userFile": str(tmp_path / "u.md"),
            "memoryFile": str(tmp_path / "m.md"),
            "sessionFile": str(tmp_path / "s.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "sa"),
            "archivedPath": str(tmp_path / "sr"),
            "registryPath": str(tmp_path / "r.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "k.json")},
        "sessions": {"storePath": str(tmp_path / "sess")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_config(path)
    assert config.vision.enabled is True
    assert config.vision.chat_handler == "llava-1-6"
    assert config.vision.max_tokens == 256


def test_transcription_config_defaults_and_load(tmp_path) -> None:
    import yaml

    from app.core.config import TranscriptionConfig, load_config

    assert TranscriptionConfig().model_size == "small"

    data = {
        "model": {"chatModelPath": "x.gguf", "embeddingModelPath": "e.gguf"},
        "transcription": {"modelSize": "medium", "device": "cpu", "language": "en"},
        "rag": {"dbPath": str(tmp_path / "c"), "docsPath": str(tmp_path / "d")},
        "memory": {
            "userFile": str(tmp_path / "u.md"),
            "memoryFile": str(tmp_path / "m.md"),
            "sessionFile": str(tmp_path / "s.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "sa"),
            "archivedPath": str(tmp_path / "sr"),
            "registryPath": str(tmp_path / "r.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "k.json")},
        "sessions": {"storePath": str(tmp_path / "sess")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_config(path)
    assert config.transcription.model_size == "medium"
    assert config.transcription.device == "cpu"
    assert config.transcription.language == "en"
