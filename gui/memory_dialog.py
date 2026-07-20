"""Memory dialog — view and edit the three memory sections inline.

Reads `memory data` (raw content + char limits) and saves via `memory-set`.
Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from gui.api_client import ApiClient

SECTIONS = ["user", "memory", "session"]
SECTION_LABELS = {
    "user": "User facts (user.md)",
    "memory": "Project memory (memory.md)",
    "session": "Session notes (session.md)",
}


class MemoryDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Memory")
        self.resize(620, 520)
        self._data: dict = {}
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.injection_label = QLabel()
        layout.addWidget(self.injection_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Section:"))
        self.section_combo = QComboBox()
        self.section_combo.addItems([SECTION_LABELS[s] for s in SECTIONS])
        self.section_combo.currentIndexChanged.connect(self._load_section)
        row.addWidget(self.section_combo, stretch=1)
        layout.addLayout(row)

        self.editor = QPlainTextEdit()
        layout.addWidget(self.editor, stretch=1)
        self.count_label = QLabel()
        self.editor.textChanged.connect(self._update_count)
        layout.addWidget(self.count_label)

        buttons = QHBoxLayout()
        save = QPushButton("Save section")
        save.clicked.connect(self._save)
        clear_all = QPushButton("Clear all memory")
        clear_all.clicked.connect(self._clear_all)
        forget = QPushButton("Forget conversation history")
        forget.clicked.connect(self._forget)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (save, clear_all, forget, close):
            buttons.addWidget(b)
        layout.addLayout(buttons)

    def _current_section(self) -> str:
        return SECTIONS[self.section_combo.currentIndex()]

    def _refresh(self) -> None:
        try:
            self._data = self.client.command("memory", "data").get("data", {})
        except Exception:  # noqa: BLE001
            self._data = {}
        on = self._data.get("injection_on")
        self.injection_label.setText(
            f"Memory injection: {'ON' if on else 'OFF'} "
            "(toggle in Features). Edits save to the .md files."
        )
        self._load_section()

    def _load_section(self) -> None:
        section = self._current_section()
        info = self._data.get("sections", {}).get(section, {})
        self.editor.setPlainText(info.get("content", ""))
        self._update_count()

    def _update_count(self) -> None:
        section = self._current_section()
        limit = self._data.get("sections", {}).get(section, {}).get("limit", 0)
        self.count_label.setText(f"{len(self.editor.toPlainText())} / {limit} chars")

    def _save(self) -> None:
        section = self._current_section()
        content = self.editor.toPlainText()
        result = self.client.command("memory-set", f"{section}\n{content}")
        self.injection_label.setText(result.get("text", "Saved."))
        self._refresh()

    def _clear_all(self) -> None:
        if QMessageBox.question(self, "Clear all memory",
                                "Wipe user/memory/session files AND conversation history?"
                                ) == QMessageBox.StandardButton.Yes:
            self.client.command("memory-clear")
            self._refresh()

    def _forget(self) -> None:
        if QMessageBox.question(self, "Forget history",
                                "Clear remembered conversation history (episodic memory)?"
                                ) == QMessageBox.StandardButton.Yes:
            result = self.client.command("memory-forget")
            self.injection_label.setText(result.get("text", "Forgotten."))
