"""Main SoulForge desktop window (PySide6).

Chat transcript + input on the left, a command button bar that maps to the same
commands the TUI exposes (routed through the API's /command endpoint). Streaming
replies arrive via ChatStreamWorker signals. Requires PySide6 (Windows venv).
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import html

from gui.api_client import ApiClient
from gui.settings import GuiSettings
from gui.streaming import ChatStreamWorker, CommandWorker


def _escape_html(text: str) -> str:
    """Escape command output so it renders literally inside a <pre> block."""
    return html.escape(text)

# Dialog-opener buttons: (label, dialog-open method name). Each opens a rich
# point-and-click menu so no command syntax needs to be memorized.
DIALOG_BUTTONS = [
    ("Features", "_open_features_dialog"),
    ("Memory", "_open_memory_dialog"),
    ("RAG", "_open_rag_dialog"),
    ("Sessions", "_open_sessions_dialog"),
    ("Agents", "_open_agents_dialog"),
    ("Tools", "_open_tools_dialog"),
    ("Kanban", "_open_kanban_dialog"),
    ("Curator", "_open_curator_dialog"),
]


class ChatWindow(QMainWindow):
    def __init__(self, settings: GuiSettings) -> None:
        super().__init__()
        self.settings = settings
        self.client = ApiClient(settings)
        self._workers: list = []  # keep QThread refs alive
        self._assistant_anchor: int | None = None

        self._was_ready = False
        self.setWindowTitle("SoulForge")
        self.resize(900, 640)
        self._build_ui()
        self._refresh_status()

        # Poll until the model finishes loading (the server serves immediately
        # with ready=false while it loads in the background).
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)
        self._status_timer.timeout.connect(self._poll_until_ready)
        self._status_timer.start()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        # Left: transcript + input
        left = QVBoxLayout()
        self.transcript = QTextEdit(readOnly=True)
        # Wrap long lines (and long unbroken tokens) to the widget width so the
        # transcript never overflows horizontally when the window is narrow.
        self.transcript.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.transcript.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.status_label = QLabel("⏳ Loading model — please wait…")
        self.status_label.setStyleSheet("font-weight: bold;")
        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setEnabled(False)
        self.input.setPlaceholderText("Loading model, please wait…")
        self.input.returnPressed.connect(self._send_message)
        self.send_btn = QPushButton("Send")
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self.input)
        input_row.addWidget(self.send_btn)
        left.addWidget(self.status_label)
        left.addWidget(self.transcript, stretch=1)
        left.addLayout(input_row)

        # Right: command button bar
        right = QVBoxLayout()
        right.addWidget(QLabel("Menus"))
        help_btn = QPushButton("Help…")
        help_btn.clicked.connect(self._open_help_browser)
        right.addWidget(help_btn)
        manage_btn = QPushButton("Manage Models…")
        manage_btn.clicked.connect(self._open_models_dialog)
        right.addWidget(manage_btn)
        for label, method_name in DIALOG_BUTTONS:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _=False, m=method_name: getattr(self, m)()
            )
            right.addWidget(btn)
        health_btn = QPushButton("Health")
        health_btn.clicked.connect(lambda: self._run_command("health", False))
        right.addWidget(health_btn)
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

    def _refresh_status(self) -> bool:
        try:
            info = self.client.ping()
            model = info.get("model")
            # Only allow prompting once ALL requested models finished loading —
            # a background vision/agent load holds the model lock, so a prompt
            # sent early would hang until it completes.
            can_prompt = bool(info.get("ready")) and not bool(info.get("loading"))
            if can_prompt:
                self.status_label.setText(
                    f"✅ {model} ({info.get('compute_backend')}) — ready"
                )
                self.input.setPlaceholderText("Type a message and press Enter")
                if not self._was_ready:
                    self._was_ready = True
                    self._append("System", "All models loaded — ready to chat.")
            else:
                stage = info.get("stage") or "loading"
                self.status_label.setText(f"⏳ {stage} — please wait…")
                self.input.setPlaceholderText("Loading models, please wait…")
            self.input.setEnabled(can_prompt)
            self.send_btn.setEnabled(can_prompt)
            return can_prompt
        except Exception as error:  # noqa: BLE001
            self.status_label.setText(f"⏳ Waiting for server… ({error})")
            self.input.setEnabled(False)
            self.send_btn.setEnabled(False)
            return False

    def _poll_until_ready(self) -> None:
        if self._refresh_status():
            self._status_timer.stop()
            self.input.setFocus()

    # -- chat ------------------------------------------------------------

    def _append(self, who: str, text: str) -> None:
        self.transcript.append(f"<b>{who}:</b> {text}")

    def _send_message(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        # A leading slash routes through the same commands as the TUI/CLI.
        if text.startswith("/"):
            self._append("You", text)
            self._send_command(text)
            return
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

    def _send_command(self, text: str) -> None:
        """Dispatch a typed /command through the API and show it in the transcript."""
        body = text[1:].strip()
        if not body:
            return
        parts = body.split(maxsplit=1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        worker = CommandWorker(self.client, name, args)
        worker.done.connect(lambda result: self._on_typed_command_result(name, result))
        worker.error.connect(lambda t: self._append("Error", t))
        worker.finished.connect(lambda: self._drop_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _on_typed_command_result(self, name: str, result: dict) -> None:
        text = result.get("text", "") or "(no output)"
        # pre-wrap keeps command whitespace/newlines but still wraps long lines
        # (a plain <pre> would overflow horizontally on a narrow window).
        self.transcript.append(f"<b>/{name}:</b>")
        self.transcript.append(
            f'<div style="white-space: pre-wrap; font-family: monospace;">'
            f"{_escape_html(text)}</div>"
        )
        self._scroll_to_end()
        if name in ("features", "model", "models", "rag", "reload-soul"):
            self._refresh_status()

    def _scroll_to_end(self) -> None:
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

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

    def _open_help_browser(self) -> None:
        from gui.help_browser import HelpBrowserDialog

        dialog = HelpBrowserDialog(self.client, self)
        dialog.insert_requested.connect(self._insert_command)
        dialog.exec()

    def _insert_command(self, usage: str) -> None:
        # Drop the command usage into the input so the user can fill in args.
        self.input.setText(usage)
        self.input.setFocus()

    def _open_models_dialog(self) -> None:
        from gui.models_dialog import ModelsDialog

        ModelsDialog(self.client, self).exec()
        self._refresh_status()

    # -- domain menu dialogs ---------------------------------------------

    def _open_features_dialog(self) -> None:
        from gui.features_dialog import FeaturesDialog

        FeaturesDialog(self.client, self).exec()

    def _open_memory_dialog(self) -> None:
        from gui.memory_dialog import MemoryDialog

        MemoryDialog(self.client, self).exec()

    def _open_sessions_dialog(self) -> None:
        from gui.sessions_dialog import SessionsDialog

        SessionsDialog(self.client, self).exec()

    def _open_rag_dialog(self) -> None:
        from gui.rag_dialog import RagDialog

        RagDialog(self.client, self).exec()

    def _open_tools_dialog(self) -> None:
        from gui.tools_dialog import ToolsDialog

        ToolsDialog(self.client, self).exec()

    def _open_kanban_dialog(self) -> None:
        from gui.kanban_dialog import KanbanDialog

        KanbanDialog(self.client, self).exec()

    def _open_curator_dialog(self) -> None:
        from gui.curator_dialog import CuratorDialog

        CuratorDialog(self.client, self).exec()

    def _open_agents_dialog(self) -> None:
        from gui.agents_dialog import AgentsDialog

        AgentsDialog(self.client, self).exec()

    def _show_result(self, title: str, text: str) -> None:
        # Scrollable, selectable dialog so long output (e.g. /help) is fully
        # readable instead of being clipped by a message box.
        dialog = QDialog(self)
        dialog.setWindowTitle(f"/{title}")
        dialog.resize(680, 520)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit(readOnly=True)
        view.setPlainText(text or "(no output)")
        view.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(view)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

    def _drop_worker(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    # -- snapshot (Phase 2) ----------------------------------------------

    def trigger_snapshot(self) -> None:
        # Hide the app window first, then grab the screen after it has settled so
        # SoulForge's own window is never part of the capture.
        self.hide()
        QTimer.singleShot(200, self._launch_region_selector)

    def _launch_region_selector(self) -> None:
        from gui.snapshot import RegionSelector

        selector = RegionSelector()
        selector.region_selected.connect(self._on_region_selected)
        selector.cancelled.connect(self.show)
        self._region_selector = selector  # keep a reference
        selector.show()
        selector.raise_()
        selector.activateWindow()

    def _on_region_selected(self, png: bytes) -> None:
        from gui.snapshot import SnapshotWorker

        self.show()
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
