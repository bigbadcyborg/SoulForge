"""Tools dialog — toggle tools, browse the catalog, manage the shell allowlist,
test a tool, and resolve pending approvals.

Reads `tools data`; actions go through commands. Tool tests run off the UI
thread. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from gui.api_client import ApiClient
from gui.streaming import CommandWorker

# Example args to prefill the tester per tool.
EXAMPLE_ARGS = {
    "read_file": '{"path": "docs/example.md"}',
    "list_dir": '{"path": "docs"}',
    "search_docs": '{"query": "search terms"}',
    "write_file": '{"path": "app/memory/note.md", "content": "hello"}',
    "run_command": '{"command": "git status"}',
    "fetch_url": '{"url": "https://example.com"}',
    "create_task": '{"title": "My task", "description": "", "column": "backlog"}',
    "update_memory": '{"section": "user", "proposed_content": "facts", "rationale": "x"}',
    "create_skill": '{"name": "my_skill", "trigger": "", "procedure": "", "validation": ""}',
}


class ToolsDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Tools")
        self.resize(700, 640)
        self._workers: list = []
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.enabled_check = QCheckBox("Tools enabled")
        self.enabled_check.toggled.connect(self._toggle_enabled)
        layout.addWidget(self.enabled_check)
        self.flags_label = QLabel()
        layout.addWidget(self.flags_label)

        # Tool catalog
        layout.addWidget(QLabel("<b>Available tools</b>"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Tool", "Risk", "Available"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, stretch=1)

        # Tool tester
        layout.addWidget(QLabel("<b>Test a tool</b>"))
        test_row = QHBoxLayout()
        self.tool_combo = QComboBox()
        self.tool_combo.currentTextChanged.connect(self._prefill_args)
        self.args_edit = QLineEdit()
        run = QPushButton("Run test")
        run.clicked.connect(self._run_test)
        test_row.addWidget(self.tool_combo)
        test_row.addWidget(self.args_edit, stretch=1)
        test_row.addWidget(run)
        layout.addLayout(test_row)
        self.test_output = QPlainTextEdit(readOnly=True)
        self.test_output.setMaximumHeight(120)
        layout.addWidget(self.test_output)

        # Shell allowlist
        allow_row = QHBoxLayout()
        add_shell = QPushButton("Add shell allowlist entry…")
        add_shell.clicked.connect(self._add_shell)
        remove_shell = QPushButton("Remove shell entry…")
        remove_shell.clicked.connect(self._remove_shell)
        log_btn = QPushButton("View tool log")
        log_btn.clicked.connect(self._view_log)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (add_shell, remove_shell, log_btn, close):
            allow_row.addWidget(b)
        layout.addLayout(allow_row)

    def _refresh(self) -> None:
        try:
            data = self.client.command("tools", "data").get("data", {})
        except Exception:  # noqa: BLE001
            data = {}
        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(bool(data.get("tools_enabled")))
        self.enabled_check.blockSignals(False)
        self.flags_label.setText(
            f"allowShell: {data.get('allow_shell')} · allowWrite: "
            f"{data.get('allow_write')} · allowlist: "
            f"{', '.join(data.get('allowlist', [])) or '(empty)'} · pending: "
            f"{data.get('pending_count', 0)}"
        )
        defs = data.get("tool_defs", [])
        self.table.setRowCount(len(defs))
        self.tool_combo.blockSignals(True)
        self.tool_combo.clear()
        for row, d in enumerate(defs):
            self.table.setItem(row, 0, QTableWidgetItem(d.get("name", "")))
            self.table.setItem(row, 1, QTableWidgetItem(d.get("risk", "")))
            self.table.setItem(
                row, 2, QTableWidgetItem("yes" if d.get("available") else "no")
            )
            self.tool_combo.addItem(d.get("name", ""))
        self.tool_combo.blockSignals(False)
        self._prefill_args(self.tool_combo.currentText())

    def _toggle_enabled(self, state: bool) -> None:
        self.client.command("features", f"tools {'on' if state else 'off'}")
        self._refresh()

    def _prefill_args(self, tool: str) -> None:
        self.args_edit.setText(EXAMPLE_ARGS.get(tool, "{}"))

    def _run_test(self) -> None:
        tool = self.tool_combo.currentText().strip()
        args = self.args_edit.text().strip() or "{}"
        if not tool:
            return
        self.test_output.setPlainText("Running…")
        worker = CommandWorker(self.client, "tools", f"test {tool} {args}")
        worker.done.connect(lambda r: self.test_output.setPlainText(r.get("text", "")))
        worker.error.connect(lambda t: self.test_output.setPlainText(t))
        worker.finished.connect(lambda: self._drop(worker))
        self._workers.append(worker)
        worker.start()

    def _add_shell(self) -> None:
        cmd, ok = QInputDialog.getText(self, "Add shell entry", "Command prefix:")
        if ok and cmd.strip():
            self.client.command("tools", f"add-shell {cmd.strip()}")
            self._refresh()

    def _remove_shell(self) -> None:
        cmd, ok = QInputDialog.getText(self, "Remove shell entry", "Command prefix:")
        if ok and cmd.strip():
            self.client.command("tools", f"remove-shell {cmd.strip()}")
            self._refresh()

    def _view_log(self) -> None:
        text = self.client.command("tools-log").get("text", "")
        self.test_output.setPlainText(text or "(no tool events)")

    def _drop(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
