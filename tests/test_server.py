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

    def create_vision_completion(self, image_bytes: bytes, prompt: str) -> str:
        self.calls.append((image_bytes, prompt))
        return f"vision saw {len(image_bytes)} bytes: {prompt or 'describe'}"


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
        self.features = FakeFeatures({"streaming": streaming, "show_sources": False})
        self.runtime = FakeRuntime()
        self.messages: list[dict] = []
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
