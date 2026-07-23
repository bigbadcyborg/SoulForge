"""Agents dialog — enable, run a multi-agent goal, view task/checkpoint status.

Reads `agents data` (structured run); running/resuming go off the UI thread;
approve/reject/cancel/edit go through commands. Live per-line progress would
need streaming (out of scope) — the tree refreshes on completion / Refresh.
Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
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
        self.load_btn = QPushButton("Load Agents")
        self.load_btn.setToolTip(
            "Load the planner model now so clicking Run does not stall while a "
            "large model loads."
        )
        self.load_btn.clicked.connect(self._load_models)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.goal_edit, stretch=1)
        run_row.addWidget(self.load_btn)
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

        # Transient feedback ("cancelled", "cannot resume") gets its own line so
        # it never evicts the run output below it.
        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        layout.addWidget(QLabel("Run output"))
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
            self.message.setText(f"Lost contact with server: {error}")
            self._poll.stop()
            self.run_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            return
        self._apply_state(state)
        if state.get("running"):
            return

        self._poll.stop()
        self.run_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        # One authoritative re-fetch on the running -> finished transition, so a
        # completed run always shows its answer without a manual Refresh. The
        # server now reads its flags before the run record, but an older server
        # can still hand back a record written just before the answer landed.
        self._refresh()
        if not self.log.toPlainText().strip() and state.get("result"):
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
                item.setData(0, Qt.ItemDataRole.UserRole + 1, c)
        # Authoritative: the pane shows this run's output, so a previous run's
        # answer never lingers next to a new run's task list.
        self.log.setPlainText(run.get("final_answer", ""))

    def _toggle_enabled(self, state: bool) -> None:
        self.client.command("agents", "on" if state else "off")
        self._refresh()

    def _load_models(self) -> None:
        """Pay the model-load cost up front rather than inside the first run."""
        self.load_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.status.setText("⏳ loading agent models… this can take a few minutes.")
        worker = CommandWorker(self.client, "agents", "load")
        worker.done.connect(lambda r: self.message.setText(r.get("text", "")))
        worker.error.connect(lambda t: self.message.setText(f"Load failed: {t}"))
        worker.finished.connect(lambda: self._load_finished(worker))
        self._workers.append(worker)
        worker.start()

    def _load_finished(self, worker) -> None:
        self._drop(worker)
        self.load_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.status.setText("Agent models loaded. Run a goal, or select a saved run below.")

    def _drop(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def _run(self) -> None:
        goal = self.goal_edit.text().strip()
        if not goal:
            return
        try:
            resp = self.client.agents_start(goal)
        except Exception as error:  # noqa: BLE001
            self.message.setText(f"Could not start run: {error}")
            return
        if not resp.get("started"):
            self.message.setText(resp.get("message", "Could not start run."))
            return
        self.message.setText("")
        self.log.setPlainText("")
        self.status.setText("⏳ starting…")
        self.run_btn.setEnabled(False)
        # Loading a profile mid-run would fight the run for the runtime lock and
        # could evict the models it is using.
        self.load_btn.setEnabled(False)
        self._poll.start()

    def _selected_checkpoint(self) -> str:
        item = self.tree.currentItem()
        if item is None:
            return ""
        return item.data(0, Qt.ItemDataRole.UserRole) or ""

    def _checkpoint_action(self, action: str) -> None:
        cid = self._selected_checkpoint()
        if not cid:
            self.message.setText("Select a pending checkpoint first.")
            return
        if action == "approve" and not self._ensure_tool_permissions():
            return  # user declined to grant the permission
        result = self.client.command("agents", f"{action} {cid}")
        self.message.setText(result.get("text", ""))
        self._refresh()

    def _selected_checkpoint_data(self) -> dict:
        item = self.tree.currentItem()
        if item is None:
            return {}
        return item.data(0, Qt.ItemDataRole.UserRole + 1) or {}

    def _ensure_tool_permissions(self) -> bool:
        """Prompt to grant the permissions this tool call needs before running it.

        Returns True when it is safe to proceed (already permitted, or the user
        granted it just now); False if the user declined.
        """
        checkpoint = self._selected_checkpoint_data()
        call = checkpoint.get("tool_call", {}) or {}
        tool = call.get("name", "")
        if not tool:
            return True
        try:
            tools = self.client.command("tools", "data").get("data", {})
        except Exception:  # noqa: BLE001
            return True  # can't check; let the server enforce it

        needed: list[str] = []          # human-readable grants
        actions: list[tuple[str, str]] = []  # (command, args) to apply
        if not tools.get("tools_enabled"):
            needed.append("• Enable the Tools feature")
            actions.append(("features", "tools on"))

        command = str(call.get("args", {}).get("command", "")).strip()
        if tool == "run_command":
            if not tools.get("allow_shell"):
                needed.append("• Allow shell commands (tools.allowShell)")
                actions.append(("tools", "allow shell on"))
            allowlist = tools.get("allowlist", [])
            if command and not any(command.startswith(p) for p in allowlist):
                needed.append(f"• Add “{command}” to the shell allowlist")
                actions.append(("tools", f"add-shell {command}"))
        elif tool == "write_file" and not tools.get("allow_write"):
            needed.append("• Allow file writes (tools.allowWrite)")
            actions.append(("tools", "allow write on"))
        elif tool == "fetch_url" and not tools.get("allow_network"):
            needed.append("• Allow network access (tools.allowNetwork)")
            actions.append(("tools", "allow network on"))

        if not needed:
            return True

        detail = command or json.dumps(call.get("args", {}))
        answer = QMessageBox.question(
            self,
            "Grant tool permission?",
            f"The agent wants to run <b>{tool}</b>:<br><br>"
            f"<code>{detail}</code><br><br>"
            "That needs permissions you haven't granted:<br><br>"
            + "<br>".join(needed)
            + "<br><br>Grant these and approve? They are saved to config.yaml "
            "and stay enabled until you turn them off.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.message.setText("Permission not granted — checkpoint left pending.")
            return False
        for name, args in actions:
            self.client.command(name, args)
        return True

    def _resume(self) -> None:
        try:
            resp = self.client.agents_resume("")
        except Exception as error:  # noqa: BLE001
            self.message.setText(f"Could not resume: {error}")
            return
        if not resp.get("started"):
            self.message.setText(resp.get("message", "Could not resume."))
            return
        self.message.setText("")
        self.status.setText("⏳ resuming…")
        self.run_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self._poll.start()

    def _cancel(self) -> None:
        result = self.client.command("agents", "cancel")
        self.message.setText(result.get("text", ""))
        self._refresh()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._poll.stop()  # a run keeps going server-side; just stop polling
        super().closeEvent(event)
