"""Model-management dialog for the GUI.

Lets the user switch the chat model, view the agent role/profile routing, and
set or disable the snapshot vision model — all through the same /api/command
endpoints the TUI commands use. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from gui.api_client import ApiClient


class ModelsDialog(QDialog):
    """Manage chat, agent, and vision models."""

    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Model Manager")
        self.resize(560, 520)
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Chat model row
        layout.addWidget(QLabel("<b>Chat model</b>"))
        chat_row = QHBoxLayout()
        self.chat_combo = QComboBox()
        switch_btn = QPushButton("Switch")
        switch_btn.clicked.connect(self._switch_chat)
        chat_row.addWidget(self.chat_combo, stretch=1)
        chat_row.addWidget(switch_btn)
        layout.addLayout(chat_row)

        # Vision model row
        layout.addWidget(QLabel("<b>Vision model (snapshots)</b>"))
        self.vision_status = QLabel("…")
        self.vision_status.setWordWrap(True)
        layout.addWidget(self.vision_status)
        vis_row = QHBoxLayout()
        self.vision_model_edit = QLineEdit()
        self.vision_model_edit.setPlaceholderText("model .gguf (path or name in ./models)")
        self.vision_mmproj_edit = QLineEdit()
        self.vision_mmproj_edit.setPlaceholderText("mmproj .gguf")
        vis_row.addWidget(self.vision_model_edit)
        vis_row.addWidget(self.vision_mmproj_edit)
        layout.addLayout(vis_row)
        vis_btn_row = QHBoxLayout()
        set_vision_btn = QPushButton("Set vision model")
        set_vision_btn.clicked.connect(self._set_vision)
        disable_vision_btn = QPushButton("Disable vision")
        disable_vision_btn.clicked.connect(self._disable_vision)
        vis_btn_row.addWidget(set_vision_btn)
        vis_btn_row.addWidget(disable_vision_btn)
        layout.addLayout(vis_btn_row)

        # Full routing view (read-only)
        layout.addWidget(QLabel("<b>Current routing</b>"))
        self.routing_view = QPlainTextEdit(readOnly=True)
        layout.addWidget(self.routing_view, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    # -- data ------------------------------------------------------------

    def _command(self, name: str, args: str = "") -> dict:
        try:
            return self.client.command(name, args)
        except Exception as error:  # noqa: BLE001
            return {"kind": "error", "text": str(error), "success": False}

    def _refresh(self) -> None:
        # Available chat models + current one.
        models = self._command("model", "list")
        available = models.get("data", {}).get("models", [])
        current = self._command("status").get("data", {}).get("model", "")
        self.chat_combo.clear()
        self.chat_combo.addItems(available)
        if current in available:
            self.chat_combo.setCurrentText(current)

        # Vision status + routing text.
        self.vision_status.setText(self._command("models", "vision").get("text", ""))
        self.routing_view.setPlainText(self._command("models").get("text", ""))

    # -- actions ---------------------------------------------------------

    def _switch_chat(self) -> None:
        name = self.chat_combo.currentText().strip()
        if name:
            self._command("models", f"chat {name}")
            self._refresh()

    def _set_vision(self) -> None:
        model = self.vision_model_edit.text().strip()
        mmproj = self.vision_mmproj_edit.text().strip()
        if not model:
            self.vision_status.setText("Enter a vision model .gguf first.")
            return
        args = f"vision {model}" + (f" {mmproj}" if mmproj else "")
        result = self._command("models", args)
        self.vision_status.setText(result.get("text", ""))
        self._refresh()

    def _disable_vision(self) -> None:
        self._command("models", "vision off")
        self._refresh()
