"""Sessions dialog — load/save/delete/summarize saved conversations.

Reads `sessions data` (structured metadata) and acts via the session commands.
Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from gui.api_client import ApiClient


class SessionsDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Sessions")
        self.resize(680, 460)
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.status = QLabel("Saved conversations — select one to load or delete.")
        layout.addWidget(self.status)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Title", "Updated", "Turns"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table, stretch=1)

        buttons = QHBoxLayout()
        load = QPushButton("Load")
        load.clicked.connect(self._load)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete)
        save = QPushButton("Save current…")
        save.clicked.connect(self._save)
        summary = QPushButton("Summarize current")
        summary.clicked.connect(self._summary)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (load, delete, save, summary, close):
            buttons.addWidget(b)
        layout.addLayout(buttons)

    def _refresh(self) -> None:
        try:
            data = self.client.command("sessions", "data").get("data", {})
        except Exception:  # noqa: BLE001
            data = {}
        sessions = data.get("sessions", [])
        self.table.setRowCount(len(sessions))
        for row, s in enumerate(sessions):
            title = QTableWidgetItem(s.get("title", "(untitled)"))
            title.setData(Qt.ItemDataRole.UserRole, s.get("id", ""))
            self.table.setItem(row, 0, title)
            self.table.setItem(row, 1, QTableWidgetItem(str(s.get("updated_at", ""))[:19]))
            self.table.setItem(row, 2, QTableWidgetItem(str(s.get("turn_count", 0))))

    def _selected_id(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def _load(self) -> None:
        sid = self._selected_id()
        if not sid:
            return
        result = self.client.command("session-load", sid)
        self.status.setText(result.get("text", ""))

    def _delete(self) -> None:
        sid = self._selected_id()
        if not sid:
            return
        if QMessageBox.question(self, "Delete session", "Delete this saved session?"
                                ) == QMessageBox.StandardButton.Yes:
            self.client.command("sessions", f"delete {sid}")
            self._refresh()

    def _save(self) -> None:
        title, ok = QInputDialog.getText(self, "Save session", "Title (optional):")
        if not ok:
            return
        result = self.client.command("session-save", title.strip())
        self.status.setText(result.get("text", ""))
        self._refresh()

    def _summary(self) -> None:
        result = self.client.command("session-summary")
        self.status.setText(result.get("text", ""))
