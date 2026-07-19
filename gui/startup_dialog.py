"""Startup model-selection dialog.

Shown once before the main window. The user picks the session's chat model and
can optionally configure + preload the agent role models and the vision model,
so nothing has to load lazily mid-session. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.api_client import ApiClient

INHERIT = "(inherit chat model)"
AGENT_ROLES = [
    "orchestrator",
    "researcher",
    "creator",
    "executor",
    "critic",
    "synthesizer",
]


class StartupDialog(QDialog):
    """Collects the session's chat model + optional agent-role and vision setup."""

    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("SoulForge — Start Session")
        self.resize(520, 620)
        self._info = self._load_info()
        self._role_combos: dict[str, QComboBox] = {}
        self._build_ui()

    def _load_info(self) -> dict:
        try:
            return self.client.command("models", "info").get("data", {})
        except Exception:  # noqa: BLE001
            return {}

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.addWidget(
            QLabel(
                "Choose the models for this session. Loading happens up front so "
                "they're ready before you start."
            )
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        available = self._info.get("available", [])

        # Chat model
        chat_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(available)
        current = self._info.get("chat_model", "")
        if current in available:
            self.model_combo.setCurrentText(current)
        chat_form.addRow("Chat model:", self.model_combo)
        layout.addLayout(chat_form)

        # Agent roles (checkable group = preload agents)
        self.agents_group = QGroupBox("Configure && preload agent role models")
        self.agents_group.setCheckable(True)
        self.agents_group.setChecked(False)
        roles_form = QFormLayout(self.agents_group)
        role_labels = self._info.get("roles", {})
        for role in AGENT_ROLES:
            combo = QComboBox()
            combo.addItem(INHERIT)
            combo.addItems(available)
            label = role_labels.get(role, "")
            if label in available:
                combo.setCurrentText(label)
            self._role_combos[role] = combo
            roles_form.addRow(f"{role}:", combo)
        layout.addWidget(self.agents_group)

        # Vision (checkable group = preload vision)
        self.vision_group = QGroupBox("Configure && preload vision model (snapshots)")
        self.vision_group.setCheckable(True)
        self.vision_group.setChecked(False)
        vision_form = QFormLayout(self.vision_group)
        self.vision_model_combo = QComboBox()
        self.vision_model_combo.addItems(available)
        self.vision_mmproj_combo = QComboBox()
        self.vision_mmproj_combo.addItems(available)
        vinfo = self._info.get("vision", {})
        if vinfo.get("model") in available:
            self.vision_model_combo.setCurrentText(vinfo["model"])
        if vinfo.get("mmproj") in available:
            self.vision_mmproj_combo.setCurrentText(vinfo["mmproj"])
        vision_form.addRow("Vision model:", self.vision_model_combo)
        vision_form.addRow("mmproj (projector):", self.vision_mmproj_combo)
        layout.addWidget(self.vision_group)

        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # -- results ---------------------------------------------------------

    def choices(self) -> dict:
        role_models = {}
        if self.agents_group.isChecked():
            for role, combo in self._role_combos.items():
                role_models[role] = combo.currentText()
        return {
            "chat_model": self.model_combo.currentText().strip() or None,
            "load_agents": self.agents_group.isChecked(),
            "role_models": role_models,
            "load_vision": self.vision_group.isChecked(),
            "vision_model": self.vision_model_combo.currentText().strip(),
            "vision_mmproj": self.vision_mmproj_combo.currentText().strip(),
        }

    def apply(self) -> None:
        """Persist the role/vision selections via the API before loading."""
        choices = self.choices()
        if choices["load_agents"]:
            for role, model in choices["role_models"].items():
                arg = "inherit" if model == INHERIT else model
                self.client.command("models", f"role {role} {arg}")
        if choices["load_vision"] and choices["vision_model"]:
            args = f"vision {choices['vision_model']}"
            if choices["vision_mmproj"]:
                args += f" {choices['vision_mmproj']}"
            self.client.command("models", args)
