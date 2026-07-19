"""Tests for the GUI's Qt-free client layer (api_client + settings).

The Qt widgets need a display and PySide6, so they are manual-test; the API
client is exercised in-process against the FastAPI app via TestClient.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.server.api import create_app
from gui.api_client import ApiClient
from gui.settings import GuiSettings


class _Ctrl:
    def __init__(self, token: str = "") -> None:
        self.config = SimpleNamespace(
            server=SimpleNamespace(auth_token=token),
            transcription=SimpleNamespace(
                model_size="small", device="cpu", compute_type="int8", language=""
            ),
        )
        self.model_name = "m.gguf"
        self.loaded = True
        self.compute_backend = "GPU"
        self.runtime = SimpleNamespace(vision_loaded=False)

    def run_health_check(self) -> str:
        return "healthy"


def _client(token: str = "") -> ApiClient:
    app = create_app(_Ctrl(token))
    test_client = TestClient(app)
    settings = GuiSettings(auth_token=token)
    return ApiClient(settings, http_client=test_client)


def test_settings_urls() -> None:
    s = GuiSettings(host="127.0.0.1", port=8765, auth_token="")
    assert s.base_url == "http://127.0.0.1:8765"
    assert s.ws_url == "ws://127.0.0.1:8765/ws/chat"
    s_tok = GuiSettings(auth_token="abc")
    assert s_tok.ws_url.endswith("/ws/chat?token=abc")


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SOULFORGE_PORT", "9999")
    monkeypatch.setenv("SOULFORGE_TOKEN", "zzz")
    s = GuiSettings.from_env()
    assert s.port == 9999
    assert s.auth_token == "zzz"


def test_api_client_ping_and_command() -> None:
    client = _client()
    assert client.ping()["model"] == "m.gguf"
    result = client.command("health")
    assert result["kind"] == "message"
    assert "healthy" in result["text"]


def test_api_client_commands_list() -> None:
    client = _client()
    names = client.commands()
    assert "health" in names


def test_api_client_sends_auth_header() -> None:
    client = _client(token="secret")
    # Correct token flows through the injected header.
    assert client.command("health")["success"] is True


# -- Qt-free GUI helpers -------------------------------------------------


def test_normalize_rect_orders_corners() -> None:
    from gui.util import normalize_rect

    # bottom-right to top-left drag still yields a positive-size rect
    assert normalize_rect(100, 100, 40, 30) == (40, 30, 60, 70)
    assert normalize_rect(10, 10, 50, 60) == (10, 10, 40, 50)


def test_to_pynput_hotkey_format() -> None:
    from gui.util import to_pynput_hotkey

    assert to_pynput_hotkey("ctrl+alt+s") == "<ctrl>+<alt>+s"
    assert to_pynput_hotkey("Ctrl+Shift+A") == "<ctrl>+<shift>+a"


def test_frames_to_wav_bytes() -> None:
    import pytest

    pytest.importorskip("soundfile")  # Windows GUI-side dependency
    import numpy as np

    from gui.util import frames_to_wav_bytes

    frames = [np.zeros((1600, 1), dtype="float32") for _ in range(3)]
    wav = frames_to_wav_bytes(frames, sample_rate=16000)
    assert wav[:4] == b"RIFF"  # valid WAV header
    # Empty recording still yields a valid (silent) WAV.
    assert frames_to_wav_bytes([], 16000)[:4] == b"RIFF"
