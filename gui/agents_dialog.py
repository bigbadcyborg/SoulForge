"""Agents dialog — enable, run a multi-agent goal, view task/checkpoint status.

Reads `agents data` (structured run); running/resuming go off the UI thread;
approve/reject/cancel/edit go through commands. Live per-line progress would
need streaming (out of scope) — the tree refreshes on completion / Refresh.
Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from gui.api_client import ApiClient
from gui.streaming import CommandWorker


class AgentsDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Multi-Agent Workflows")
        self.resize(720, 620)
        self._workers: list = []
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.enabled_check = QCheckBox("Agents enabled")
        self.enabled_check.toggled.connect(self._toggle_enabled)
        layout.addWidget(self.enabled_check)

        run_row = QHBoxLayout()
        self.goal_edit = QLineEdit()
        self.goal_edit.setPlaceholderText("Goal for the agent run…")
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.goal_edit, stretch=1)
        run_row.addWidget(self.run_btn)
        layout.addLayout(run_row)

        self.status = QLabel("Run a goal, or select a saved run below.")
        layout.addWidget(self.status)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Task / checkpoint", "Status"])
        layout.addWidget(self.tree, stretch=1)

        # Checkpoint / run controls
        ck = QHBoxLayout()
        approve = QPushButton("Approve checkpoint")
        approve.clicked.connect(lambda: self._checkpoint_action("approve"))
        reject = QPushButton("Reject checkpoint")
        reject.clicked.connect(lambda: self._checkpoint_action("reject"))
        resume = QPushButton("Resume run")
        resume.clicked.connect(self._resume)
        cancel = QPushButton("Cancel run")
        cancel.clicked.connect(self._cancel)
        for b in (approve, reject, resume, cancel):
            ck.addWidget(b)
        layout.addLayout(ck)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

        bottom = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        bottom.addWidget(refresh)
        bottom.addWidget(close)
        layout.addLayout(bottom)

    def _refresh(self) -> None:
        try:
            data = self.client.command("agents", "data").get("data", {})
        except Exception:  # noqa: BLE001
            data = {}
        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(bool(data.get("enabled")))
        self.enabled_check.blockSignals(False)
        self._current = data.get("current")
        self.tree.clear()
        if not self._current:
            self.status.setText("No agent runs yet.")
            return
        run = self._current
        self.status.setText(
            f"Run {run.get('run_id', '')[:12]} — {run.get('status', '')}: {run.get('goal', '')}"
        )
        tasks_node = QTreeWidgetItem(self.tree, ["Tasks", ""])
        tasks_node.setExpanded(True)
        for task in run.get("tasks", []):
            QTreeWidgetItem(
                tasks_node,
                [f"{task.get('role', '')}/{task.get('id', '')}: {task.get('title', '')}",
                 task.get("status", "")],
            )
        pending = [c for c in run.get("checkpoints", []) if c.get("status") == "pending"]
        if pending:
            ck_node = QTreeWidgetItem(self.tree, ["Pending checkpoints", ""])
            ck_node.setExpanded(True)
            for c in pending:
                item = QTreeWidgetItem(
                    ck_node,
                    [f"{c.get('checkpoint_id', '')}: {c.get('tool_call', {}).get('name', 'tool')}",
                     c.get("risk", "")],
                )
                item.setData(0, Qt.ItemDataRole.UserRole, c.get("checkpoint_id", ""))
        if run.get("final_answer"):
            self.log.setPlainText(run["final_answer"])

    def _toggle_enabled(self, state: bool) -> None:
        self.client.command("agents", "on" if state else "off")
        self._refresh()

    def _run(self) -> None:
        goal = self.goal_edit.text().strip()
        if not goal:
            return
        self.status.setText("Running agent workflow… (this can take a while)")
        self.run_btn.setEnabled(False)
        worker = CommandWorker(self.client, "agents", f"run {goal}")
        worker.done.connect(self._on_run_done)
        worker.error.connect(lambda t: self._on_run_done({"text": t}))
        worker.finished.connect(lambda: self._drop(worker))
        self._workers.append(worker)
        worker.start()

    def _on_run_done(self, result: dict) -> None:
        self.run_btn.setEnabled(True)
        self.log.setPlainText(result.get("text", ""))
        self._refresh()

    def _selected_checkpoint(self) -> str:
        item = self.tree.currentItem()
        if item is None:
            return ""
        return item.data(0, Qt.ItemDataRole.UserRole) or ""

    def _checkpoint_action(self, action: str) -> None:
        cid = self._selected_checkpoint()
        if not cid:
            self.status.setText("Select a pending checkpoint first.")
            return
        result = self.client.command("agents", f"{action} {cid}")
        self.log.setPlainText(result.get("text", ""))
        self._refresh()

    def _resume(self) -> None:
        self.status.setText("Resuming…")
        worker = CommandWorker(self.client, "agents", "resume")
        worker.done.connect(self._on_run_done)
        worker.error.connect(lambda t: self._on_run_done({"text": t}))
        worker.finished.connect(lambda: self._drop(worker))
        self._workers.append(worker)
        worker.start()

    def _cancel(self) -> None:
        result = self.client.command("agents", "cancel")
        self.log.setPlainText(result.get("text", ""))
        self._refresh()

    def _drop(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
