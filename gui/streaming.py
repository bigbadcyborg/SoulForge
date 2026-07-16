"""Background chat streaming for the Qt GUI.

A ``QThread`` consumes the ``/ws/chat`` WebSocket and re-emits each frame as a Qt
signal, so widget updates happen on the GUI thread (the analog of Textual's
``call_from_thread``). Import requires PySide6 (Windows GUI venv only).
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gui.api_client import ApiClient, stream_chat


class ChatStreamWorker(QThread):
    """Streams one assistant reply; emits frames as typed signals."""

    token = Signal(str)
    final = Signal(str)
    review = Signal(str)
    sources = Signal(list)
    tool = Signal(list)
    error = Signal(str)

    def __init__(self, ws_url: str, message: str) -> None:
        super().__init__()
        self._ws_url = ws_url
        self._message = message

    def run(self) -> None:
        try:
            for frame in stream_chat(self._ws_url, self._message):
                kind = frame.get("type")
                if kind == "token":
                    self.token.emit(frame.get("token", ""))
                elif kind == "final":
                    self.final.emit(frame.get("text", ""))
                elif kind == "review":
                    self.review.emit(frame.get("text", ""))
                elif kind == "sources":
                    self.sources.emit(frame.get("sources", []))
                elif kind == "tool":
                    self.tool.emit(frame.get("pending", []))
                elif kind == "error":
                    self.error.emit(frame.get("text", "Unknown error"))
        except Exception as error:  # noqa: BLE001 - surface connection failures
            self.error.emit(f"Connection error: {error}")


class CommandWorker(QThread):
    """Runs one REST command off the UI thread; emits the result dict."""

    done = Signal(dict)
    error = Signal(str)

    def __init__(self, client: ApiClient, name: str, args: str = "") -> None:
        super().__init__()
        self._client = client
        self._name = name
        self._args = args

    def run(self) -> None:
        try:
            self.done.emit(self._client.command(self._name, self._args))
        except Exception as error:  # noqa: BLE001
            self.error.emit(f"{self._name} failed: {error}")
