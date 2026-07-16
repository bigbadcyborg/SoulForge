"""Main SoulForge desktop window (PySide6).

Chat transcript + input on the left, a command button bar that maps to the same
commands the TUI exposes (routed through the API's /command endpoint). Streaming
replies arrive via ChatStreamWorker signals. Requires PySide6 (Windows venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.api_client import ApiClient
from gui.settings import GuiSettings
from gui.streaming import ChatStreamWorker, CommandWorker

# Command buttons: (label, command name, prompt-for-args?). A prompt lets the
# user supply the sub-command/argument (e.g. "run <goal>", "load <id>").
COMMAND_BUTTONS = [
    ("Health", "health", False),
    ("Features", "features", True),
    ("Model", "model", True),
    ("Memory", "memory", False),
    ("RAG", "rag", True),
    ("Sessions", "sessions", True),
    ("Agents", "agents", True),
    ("Tools", "tools", True),
]


class ChatWindow(QMainWindow):
    def __init__(self, settings: GuiSettings) -> None:
        super().__init__()
        self.settings = settings
        self.client = ApiClient(settings)
        self._workers: list = []  # keep QThread refs alive
        self._assistant_anchor: int | None = None

        self.setWindowTitle("SoulForge")
        self.resize(900, 640)
        self._build_ui()
        self._refresh_status()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        # Left: transcript + input
        left = QVBoxLayout()
        self.transcript = QTextEdit(readOnly=True)
        self.status_label = QLabel("Connecting...")
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message and press Enter")
        self.input.returnPressed.connect(self._send_message)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self.input)
        input_row.addWidget(self.send_btn)
        left.addWidget(self.status_label)
        left.addWidget(self.transcript, stretch=1)
        left.addLayout(input_row)

        # Right: command button bar
        right = QVBoxLayout()
        right.addWidget(QLabel("Commands"))
        for label, name, prompt in COMMAND_BUTTONS:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _=False, n=name, p=prompt: self._run_command(n, p)
            )
            right.addWidget(btn)
        right.addStretch(1)
        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.clicked.connect(self.trigger_snapshot)
        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.clicked.connect(self.trigger_transcribe)
        right.addWidget(self.snapshot_btn)
        right.addWidget(self.transcribe_btn)

        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=1)
        self.setCentralWidget(root)

    # -- status ----------------------------------------------------------

    def _refresh_status(self) -> None:
        try:
            info = self.client.ping()
            ready = info.get("ready")
            self.status_label.setText(
                f"Model: {info.get('model')} ({info.get('compute_backend')}) — "
                f"{'ready' if ready else 'loading...'}"
            )
            self.input.setEnabled(bool(ready))
            self.send_btn.setEnabled(bool(ready))
        except Exception as error:  # noqa: BLE001
            self.status_label.setText(f"Server unreachable: {error}")
            self.input.setEnabled(False)
            self.send_btn.setEnabled(False)

    # -- chat ------------------------------------------------------------

    def _append(self, who: str, text: str) -> None:
        self.transcript.append(f"<b>{who}:</b> {text}")

    def _send_message(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._append("You", text)
        self.input.setEnabled(False)
        self.send_btn.setEnabled(False)

        self.transcript.append("<b>SoulForge:</b> ")
        self._assistant_anchor = self.transcript.textCursor().position()

        worker = ChatStreamWorker(self.settings.ws_url, text)
        worker.token.connect(self._on_token)
        worker.final.connect(self._on_final)
        worker.review.connect(lambda t: self._append("System", t))
        worker.error.connect(lambda t: self._append("Error", t))
        worker.finished.connect(lambda: self._on_stream_done(worker))
        self._workers.append(worker)
        worker.start()

    def _on_token(self, token: str) -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.transcript.setTextCursor(cursor)

    def _on_final(self, text: str) -> None:
        # Tool parsing may rewrite the displayed text; keep it simple and leave
        # the streamed text in place. A future revision can reconcile.
        pass

    def _on_stream_done(self, worker: ChatStreamWorker) -> None:
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input.setFocus()
        if worker in self._workers:
            self._workers.remove(worker)

    # -- commands --------------------------------------------------------

    def _run_command(self, name: str, prompt: bool) -> None:
        args = ""
        if prompt:
            text, ok = QInputDialog.getText(
                self, name, f"Arguments for /{name} (blank for default):"
            )
            if not ok:
                return
            args = text.strip()
        worker = CommandWorker(self.client, name, args)
        worker.done.connect(lambda result: self._on_command_result(name, result))
        worker.error.connect(lambda t: self._show_result(name, t))
        worker.finished.connect(lambda: self._drop_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_command_result(self, name: str, result: dict) -> None:
        self._show_result(name, result.get("text", "(no output)"))
        if name in ("features", "model", "models"):
            self._refresh_status()

    def _show_result(self, title: str, text: str) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"/{title}")
        dialog.setText(text or "(no output)")
        dialog.setTextInteractionFlags(Qt.TextSelectableByMouse)
        dialog.exec()

    def _drop_worker(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    # -- snapshot (Phase 2) ----------------------------------------------

    def trigger_snapshot(self) -> None:
        from gui.snapshot import RegionSelector

        self.hide()  # keep the app window out of the capture
        selector = RegionSelector()
        selector.region_selected.connect(self._on_region_selected)
        selector.cancelled.connect(self.show)
        self._region_selector = selector  # keep a reference
        selector.show()

    def _on_region_selected(self, left: int, top: int, width: int, height: int) -> None:
        from gui.snapshot import SnapshotWorker, capture_region_png

        self.show()
        try:
            png = capture_region_png(left, top, width, height)
        except Exception as error:  # noqa: BLE001
            self._show_result("snapshot", f"Capture failed: {error}")
            return
        prompt, ok = QInputDialog.getText(
            self, "Snapshot", "Ask about the capture (blank = describe):"
        )
        if not ok:
            return
        self._append("You", f"[snapshot] {prompt or 'Describe this image.'}")
        worker = SnapshotWorker(self.client, png, prompt.strip())
        worker.done.connect(lambda t: self._append("SoulForge", t))
        worker.error.connect(lambda t: self._append("Error", t))
        worker.finished.connect(lambda: self._drop_worker(worker))
        self._workers.append(worker)
        worker.start()

    # -- transcribe (Phase 3) --------------------------------------------

    def trigger_transcribe(self) -> None:
        from gui.transcribe import MicRecorder, TranscribeWorker

        recorder = getattr(self, "_recorder", None)
        if recorder is None:
            recorder = MicRecorder()
            self._recorder = recorder

        if not recorder.recording:
            try:
                recorder.start()
            except Exception as error:  # noqa: BLE001
                self._show_result("transcribe", f"Mic unavailable: {error}")
                return
            self.status_label.setText("Recording... trigger transcribe again to stop.")
            return

        wav = recorder.stop()
        self.status_label.setText("Transcribing...")
        worker = TranscribeWorker(self.client, wav)
        worker.done.connect(self._on_transcribed)
        worker.error.connect(lambda t: self._show_result("transcribe", t))
        worker.finished.connect(lambda: self._drop_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_transcribed(self, text: str) -> None:
        self._refresh_status()
        if not text:
            return
        # Insert into the input box for review rather than auto-sending.
        existing = self.input.text()
        self.input.setText((existing + " " + text).strip() if existing else text)
        self.input.setFocus()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.client.close()
        super().closeEvent(event)
