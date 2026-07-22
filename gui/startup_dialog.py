"""Startup model-selection dialog.

Shown once before the main window. The user picks the session's chat model and
can optionally configure + preload the agent role models and the vision model,
so nothing has to load lazily mid-session. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QToolButton,
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

# Shown by the "?" button next to each role: what the role does and what kind
# of model suits it. Roles pointing at the same file share one loaded copy, so
# reusing a model across roles costs no extra VRAM.
ROLE_HELP: dict[str, tuple[str, str]] = {
    "orchestrator": (
        "Breaks your goal into a dependency-ordered task graph and decides "
        "revisions. This is the hardest reasoning job in the pipeline and the "
        "most common place a run fails, so give it your strongest model.",
        "Type: general instruct / reasoning (not a coder model)\n"
        "Size: 30B–70B ideal · 12B workable · under 7B often fails to emit a "
        "valid task graph\n"
        "Tip: it runs only during planning, so a big 'swap' model here costs "
        "time but not sustained VRAM.",
    ),
    "researcher": (
        "Gathers scoped local context — retrieved documents, files, memory — "
        "and summarizes it for the other roles without inventing sources.",
        "Type: general instruct, good at summarizing/extraction\n"
        "Size: 7B–12B is plenty\n"
        "Tip: speed matters more than depth; a small model keeps runs snappy.",
    ),
    "creator": (
        "Builds the main deliverable — the code, document, or draft that the "
        "run is actually producing. Usually the quality bottleneck.",
        "Type: a coder model if your goal is code (e.g. Qwen2.5-Coder), "
        "otherwise a strong general instruct model\n"
        "Size: 12B–32B · larger pays off most here\n"
        "Tip: this is the best place to spend your VRAM budget after the "
        "orchestrator.",
    ),
    "executor": (
        "Requests local tools (shell, file, fetch) and reports what happened. "
        "It must emit precise JSON tool arguments rather than prose.",
        "Type: general instruct with reliable instruction-following\n"
        "Size: 7B–12B\n"
        "Tip: accuracy beats creativity; tool calls still require your "
        "approval, so a smaller model here is low-risk.",
    ),
    "critic": (
        "Checks the run's output against the goal and success criteria, and "
        "asks for revisions when it falls short.",
        "Type: general instruct, strong at following rubrics\n"
        "Size: 7B–14B\n"
        "Tip: runs at a low temperature by design — a mid-size model is fine; "
        "too small and it rubber-stamps bad output.",
    ),
    "synthesizer": (
        "Writes the final user-facing answer once the graph passes review, "
        "grounded in the completed task outputs.",
        "Type: general instruct with good writing quality\n"
        "Size: 12B–32B\n"
        "Tip: this is the text you actually read, so don't go too small.",
    ),
}


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
            roles_form.addRow(f"{role}:", self._role_row(role, combo))
        note = QLabel(
            "Roles set to the same model share one loaded copy — reusing a "
            "model across roles costs no extra VRAM."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        roles_form.addRow(note)
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

    def _role_row(self, role: str, combo: QComboBox) -> QWidget:
        """A role's model dropdown plus a clickable '?' with recommendations."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(combo, stretch=1)
        help_btn = QToolButton()
        help_btn.setText("?")
        help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        summary, recommendation = ROLE_HELP.get(role, ("", ""))
        help_btn.setToolTip(summary or f"About the {role} role")
        help_btn.clicked.connect(lambda _=False, r=role: self._show_role_help(r))
        layout.addWidget(help_btn)
        return row

    def _show_role_help(self, role: str) -> None:
        summary, recommendation = ROLE_HELP.get(
            role, ("No guidance available.", "")
        )
        box = QMessageBox(self)
        box.setWindowTitle(f"{role.capitalize()} — model guidance")
        box.setText(f"<b>What it does</b><br>{summary}")
        box.setInformativeText(f"Recommended model\n\n{recommendation}")
        box.exec()

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
