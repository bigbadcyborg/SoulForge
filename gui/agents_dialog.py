"""Agents dialog — enable, run a multi-agent goal, view task/checkpoint status.

Reads `agents data` (structured run); running/resuming go off the UI thread;
approve/reject/cancel/edit go through commands. Live per-line progress would
need streaming (out of scope) — the tree refreshes on completion / Refresh.
Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
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


class AgentsDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Multi-Agent Workflows")
        self.resize(720, 620)
        self._workers: list = []
        self._build_ui()
        # Poll while a run is in progress: agent runs take minutes (model swaps
        # + many LLM calls), so they run server-side and we poll for progress.
        self._poll = QTimer(self)
        self._poll.setInterval(2000)
        self._poll.timeout.connect(self._poll_state)
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
        self.tree.setColumnWidth(0, 460)  # task titles were being truncated
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
            state = self.client.agents_state()
        except Exception:  # noqa: BLE001
            state = {}
        self._apply_state(state)

    def _poll_state(self) -> None:
        try:
            state = self.client.agents_state()
        except Exception as error:  # noqa: BLE001
            self.status.setText(f"Lost contact with server: {error}")
            self._poll.stop()
            self.run_btn.setEnabled(True)
            return
        self._apply_state(state)
        if not state.get("running"):
            self._poll.stop()
            self.run_btn.setEnabled(True)
            if state.get("result"):
                self.log.setPlainText(state["result"])

    def _apply_state(self, state: dict) -> None:
        data = state.get("data", {})
        if state.get("running"):
            self.status.setText(f"⏳ {state.get('stage') or 'running'}…")
        self._render(data, running=bool(state.get("running")))

    def _render(self, data: dict, running: bool = False) -> None:
        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(bool(data.get("enabled")))
        self.enabled_check.blockSignals(False)
        self._current = data.get("current")
        self.tree.clear()
        if not self._current:
            if not running:
                self.status.setText("No agent runs yet.")
            return
        run = self._current
        if not running:  # while running, the stage line owns the status label
            self.status.setText(
                f"Run {run.get('run_id', '')[:12]} — {run.get('status', '')}: "
                f"{run.get('goal', '')}"
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
        try:
            resp = self.client.agents_start(goal)
        except Exception as error:  # noqa: BLE001
            self.log.setPlainText(f"Could not start run: {error}")
            return
        if not resp.get("started"):
            self.log.setPlainText(resp.get("message", "Could not start run."))
            return
        self.log.setPlainText("")
        self.status.setText("⏳ starting…")
        self.run_btn.setEnabled(False)
        self._poll.start()

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
        try:
            resp = self.client.agents_resume("")
        except Exception as error:  # noqa: BLE001
            self.log.setPlainText(f"Could not resume: {error}")
            return
        if not resp.get("started"):
            self.log.setPlainText(resp.get("message", "Could not resume."))
            return
        self.status.setText("⏳ resuming…")
        self.run_btn.setEnabled(False)
        self._poll.start()

    def _cancel(self) -> None:
        result = self.client.command("agents", "cancel")
        self.log.setPlainText(result.get("text", ""))
        self._refresh()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._poll.stop()  # a run keeps going server-side; just stop polling
        super().closeEvent(event)
