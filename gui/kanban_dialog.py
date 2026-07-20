"""Kanban dialog — a four-column board with add/move/delete + suggestions.

Reads `kanban` (structured board); actions go through the task commands.
Suggestion generation runs off the UI thread. Requires PySide6 (Windows venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
)

from gui.api_client import ApiClient
from gui.streaming import CommandWorker

COLUMNS = [
    ("backlog", "Backlog"),
    ("in_progress", "In Progress"),
    ("blocked", "Blocked"),
    ("done", "Done"),
]


class KanbanDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Kanban Board")
        self.resize(860, 520)
        self._workers: list = []
        self._lists: dict[str, QListWidget] = {}
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.status = QLabel("Right-click a task to move or delete it.")
        layout.addWidget(self.status)

        columns = QHBoxLayout()
        for key, label in COLUMNS:
            col = QVBoxLayout()
            col.addWidget(QLabel(f"<b>{label}</b>"))
            lst = QListWidget()
            lst.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            lst.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            lst.customContextMenuRequested.connect(
                lambda pos, w=lst: self._task_menu(w, pos)
            )
            self._lists[key] = lst
            col.addWidget(lst, stretch=1)
            columns.addLayout(col)
        layout.addLayout(columns, stretch=1)

        buttons = QHBoxLayout()
        add = QPushButton("Add task…")
        add.clicked.connect(self._add_task)
        suggest = QPushButton("Suggest tasks")
        suggest.clicked.connect(self._suggest)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (add, suggest, refresh, close):
            buttons.addWidget(b)
        layout.addLayout(buttons)

    def _refresh(self) -> None:
        try:
            data = self.client.command("kanban").get("data", {})
        except Exception:  # noqa: BLE001
            data = {}
        by_key = {c["key"]: c for c in data.get("columns", [])}
        for key, lst in self._lists.items():
            lst.clear()
            for task in by_key.get(key, {}).get("tasks", []):
                item = QListWidgetItem(task.get("title", "(untitled)"))
                item.setData(Qt.ItemDataRole.UserRole, task.get("id", ""))
                item.setToolTip(task.get("description", ""))
                lst.addItem(item)

    def _task_menu(self, widget: QListWidget, pos) -> None:
        item = widget.itemAt(pos)
        if item is None:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        for key, label in COLUMNS:
            menu.addAction(f"Move to {label}", lambda k=key: self._move(task_id, k))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self._delete(task_id))
        menu.exec(widget.mapToGlobal(pos))

    def _add_task(self) -> None:
        title, ok = QInputDialog.getText(self, "Add task", "Task title:")
        if ok and title.strip():
            self.client.command("task-new", title.strip())
            self._refresh()

    def _move(self, task_id: str, column: str) -> None:
        self.client.command("task-move", f"{task_id} {column}")
        self._refresh()

    def _delete(self, task_id: str) -> None:
        self.client.command("task-delete", task_id)
        self._refresh()

    def _suggest(self) -> None:
        self.status.setText("Generating suggestions…")
        worker = CommandWorker(self.client, "task-suggest", "")
        worker.done.connect(lambda r: self.status.setText(r.get("text", "")[:200]))
        worker.error.connect(lambda t: self.status.setText(t))
        worker.finished.connect(lambda: (self._drop(worker), self._refresh()))
        self._workers.append(worker)
        worker.start()

    def _drop(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
