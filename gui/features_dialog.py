"""Feature toggle dialog — a checkbox per feature flag.

Reads `features data` and toggles via `features <name> on|off`, so the user
never types a command. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from gui.api_client import ApiClient

# Friendlier labels for the raw feature keys.
FEATURE_LABELS = {
    "soul": "Persona (SOUL.md)",
    "rag": "RAG (document retrieval)",
    "memory": "Memory",
    "skills": "Skills",
    "curator": "Curator",
    "kanban": "Kanban board",
    "agents": "Multi-agent workflows",
    "streaming": "Streaming replies",
    "show_sources": "Show RAG sources",
    "tools": "Tools",
}


class FeaturesDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Features")
        self.resize(360, 420)
        self._checks: dict[str, QCheckBox] = {}
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Toggle features</b> (saved to config.yaml)"))
        self._box = QVBoxLayout()
        layout.addLayout(self._box)
        layout.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close)

    def _refresh(self) -> None:
        try:
            features = self.client.command("features", "data").get("data", {}).get(
                "features", {}
            )
        except Exception:  # noqa: BLE001
            features = {}
        # Clear existing checkboxes.
        while self._box.count():
            item = self._box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checks.clear()
        for key, enabled in features.items():
            cb = QCheckBox(FEATURE_LABELS.get(key, key))
            cb.setChecked(bool(enabled))
            cb.toggled.connect(lambda state, k=key: self._toggle(k, state))
            self._box.addWidget(cb)
            self._checks[key] = cb

    def _toggle(self, key: str, state: bool) -> None:
        self.client.command("features", f"{key} {'on' if state else 'off'}")
