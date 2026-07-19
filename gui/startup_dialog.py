"""Startup model-selection dialog.

Shown once before the main window: the user picks the chat model for the
session and can opt to preload the agent role models and/or the vision model so
nothing has to load lazily mid-session. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from gui.api_client import ApiClient


class StartupDialog(QDialog):
    """Collects the session's chat model + optional agent/vision preloads."""

    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("SoulForge — Start Session")
        self.resize(460, 260)
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Choose the model for this session. Loading happens up front so "
                "the model is ready before you start."
            )
        )
        form = QFormLayout()
        self.model_combo = QComboBox()
        form.addRow("Chat model:", self.model_combo)
        layout.addLayout(form)

        self.agents_check = QCheckBox(
            "Preload agent role models (for /agents workflows)"
        )
        self.vision_check = QCheckBox("Preload vision model (for snapshots)")
        layout.addWidget(self.agents_check)
        layout.addWidget(self.vision_check)

        self.vision_note = QLabel()
        self.vision_note.setWordWrap(True)
        self.vision_note.setStyleSheet("color: gray;")
        layout.addWidget(self.vision_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        # Available chat models (works before any model is loaded).
        try:
            models = self.client.command("model", "list").get("data", {}).get("models", [])
        except Exception:  # noqa: BLE001
            models = []
        self.model_combo.addItems(models)
        try:
            current = self.client.ping().get("model", "")
            if current in models:
                self.model_combo.setCurrentText(current)
        except Exception:  # noqa: BLE001
            pass

        # Vision availability note (from /models vision).
        try:
            vtext = self.client.command("models", "vision").get("text", "")
        except Exception:  # noqa: BLE001
            vtext = ""
        if "disabled" in vtext.lower():
            self.vision_check.setEnabled(False)
            self.vision_note.setText(
                "No vision model configured — set one in Manage Models to enable snapshots."
            )
        else:
            self.vision_note.setText(vtext.splitlines()[0] if vtext else "")

    def choices(self) -> dict:
        return {
            "chat_model": self.model_combo.currentText().strip() or None,
            "load_agents": self.agents_check.isChecked(),
            "load_vision": self.vision_check.isChecked(),
        }
