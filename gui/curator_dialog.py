"""Curator dialog — run a skill review, act on findings, manage skills.

Reads `curator data`; review runs off the UI thread; accept/dismiss/archive/
restore/compact go through commands. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from gui.api_client import ApiClient
from gui.streaming import CommandWorker


class CuratorDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Curator — Skill Maintenance")
        self.resize(640, 560)
        self._workers: list = []
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.status = QLabel("Run a review to find stale or bloated skills.")
        layout.addWidget(self.status)

        layout.addWidget(QLabel("<b>Findings</b>"))
        self.findings = QListWidget()
        self.findings.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.findings, stretch=1)
        fbtn = QHBoxLayout()
        review = QPushButton("Run review")
        review.clicked.connect(self._review)
        accept = QPushButton("Accept finding")
        accept.clicked.connect(self._accept)
        dismiss = QPushButton("Dismiss finding")
        dismiss.clicked.connect(self._dismiss)
        for b in (review, accept, dismiss):
            fbtn.addWidget(b)
        layout.addLayout(fbtn)

        layout.addWidget(QLabel("<b>Active skills</b>"))
        self.active = QListWidget()
        layout.addWidget(self.active)
        layout.addWidget(QLabel("<b>Archived skills</b>"))
        self.archived = QListWidget()
        layout.addWidget(self.archived)
        sbtn = QHBoxLayout()
        archive = QPushButton("Archive selected active")
        archive.clicked.connect(self._archive)
        compact = QPushButton("Compact selected active")
        compact.clicked.connect(self._compact)
        restore = QPushButton("Restore selected archived")
        restore.clicked.connect(self._restore)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (archive, compact, restore, close):
            sbtn.addWidget(b)
        layout.addLayout(sbtn)

    def _refresh(self) -> None:
        try:
            data = self.client.command("curator", "data").get("data", {})
        except Exception:  # noqa: BLE001
            data = {}
        self.findings.clear()
        for f in data.get("findings", []):
            label = (
                f"[{f.get('proposed_action', '?')}] {f.get('skill_name', '')} "
                f"({f.get('finding_type', '')}) — {f.get('rationale', '')}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, f.get("finding_id", ""))
            self.findings.addItem(item)
        self.active.clear()
        for s in data.get("active_skills", []):
            self.active.addItem(s.get("name", ""))
        self.archived.clear()
        for s in data.get("archived_skills", []):
            self.archived.addItem(s.get("name", ""))

    def _review(self) -> None:
        self.status.setText("Reviewing skills…")
        worker = CommandWorker(self.client, "curator", "")
        worker.done.connect(lambda r: self.status.setText(r.get("text", "")[:200]))
        worker.error.connect(lambda t: self.status.setText(t))
        worker.finished.connect(lambda: (self._drop(worker), self._refresh()))
        self._workers.append(worker)
        worker.start()

    def _accept(self) -> None:
        item = self.findings.currentItem()
        if item is None:
            return
        self.client.command("curator-accept", item.data(Qt.ItemDataRole.UserRole))
        self._refresh()

    def _dismiss(self) -> None:
        item = self.findings.currentItem()
        if item is None:
            return
        self.client.command("curator-ignore", item.data(Qt.ItemDataRole.UserRole))
        self._refresh()

    def _archive(self) -> None:
        item = self.active.currentItem()
        if item:
            self.client.command("curator-archive", item.text())
            self._refresh()

    def _compact(self) -> None:
        item = self.active.currentItem()
        if item:
            self.status.setText(
                self.client.command("curator-compact", item.text()).get("text", "")[:200]
            )
            self._refresh()

    def _restore(self) -> None:
        item = self.archived.currentItem()
        if item:
            self.client.command("skill-restore", item.text())
            self._refresh()

    def _drop(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
