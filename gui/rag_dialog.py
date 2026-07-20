"""RAG dialog — toggle RAG, pick active sources, add/remove documents, ingest.

Reads `rag data`; enable/disable, source selection, and remove go through
commands; adding a document uploads the file then ingests. Ingest runs off the
UI thread. Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from gui.api_client import ApiClient
from gui.streaming import CommandWorker


class RagDialog(QDialog):
    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("RAG — Documents & Retrieval")
        self.resize(560, 520)
        self._workers: list = []
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.enabled_check = QCheckBox("RAG enabled (retrieve from documents)")
        self.enabled_check.toggled.connect(self._toggle_enabled)
        layout.addWidget(self.enabled_check)

        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

        layout.addWidget(QLabel("<b>Sources</b> (checked = used for retrieval):"))
        self.sources = QListWidget()
        layout.addWidget(self.sources, stretch=1)

        src_buttons = QHBoxLayout()
        apply_sel = QPushButton("Apply selection")
        apply_sel.clicked.connect(self._apply_sources)
        select_all = QPushButton("Use all")
        select_all.clicked.connect(self._select_all)
        remove = QPushButton("Remove file")
        remove.clicked.connect(self._remove)
        for b in (apply_sel, select_all, remove):
            src_buttons.addWidget(b)
        layout.addLayout(src_buttons)

        bottom = QHBoxLayout()
        add = QPushButton("Add document…")
        add.clicked.connect(self._add_document)
        self.ingest_btn = QPushButton("Ingest / re-index")
        self.ingest_btn.clicked.connect(self._ingest)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (add, self.ingest_btn, close):
            bottom.addWidget(b)
        layout.addLayout(bottom)

    def _refresh(self) -> None:
        try:
            data = self.client.command("rag", "data").get("data", {})
        except Exception:  # noqa: BLE001
            data = {}
        status = data.get("status", {})
        stats = data.get("stats", {})
        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(bool(status.get("enabled")))
        self.enabled_check.blockSignals(False)
        self.stats_label.setText(
            f"Indexed chunks: {stats.get('chunk_count', 0)} across "
            f"{len(status.get('available_sources') or [])} document(s)."
        )
        selected = status.get("selected_sources")  # None = all
        self.sources.clear()
        for src in status.get("available_sources") or []:
            item = QListWidgetItem(src)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = selected is None or src in selected
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.sources.addItem(item)

    def _toggle_enabled(self, state: bool) -> None:
        self.client.command("rag", "on" if state else "off")
        self._refresh()

    def _checked_sources(self) -> list[str]:
        out = []
        for i in range(self.sources.count()):
            item = self.sources.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _apply_sources(self) -> None:
        checked = self._checked_sources()
        total = self.sources.count()
        if len(checked) == total:
            self.client.command("rag", "select all")
        else:
            self.client.command("rag", "select " + ",".join(checked))
        self._refresh()

    def _select_all(self) -> None:
        self.client.command("rag", "select all")
        self._refresh()

    def _remove(self) -> None:
        item = self.sources.currentItem()
        if item is None:
            return
        if QMessageBox.question(self, "Remove document",
                                f"Delete {item.text()} from docs/ and re-index?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        self.client.command("rag", f"remove {item.text()}")
        self._ingest()

    def _add_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add document",
            filter="Documents (*.txt *.md *.pdf);;All files (*.*)",
        )
        if not path:
            return
        try:
            content = Path(path).read_bytes()
            self.client.upload_doc(Path(path).name, content)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Upload failed", str(error))
            return
        if QMessageBox.question(self, "Ingest", "Document added. Index it now?"
                                ) == QMessageBox.StandardButton.Yes:
            self._ingest()

    def _ingest(self) -> None:
        self.stats_label.setText("Ingesting… (this can take a moment)")
        self.ingest_btn.setEnabled(False)
        worker = CommandWorker(self.client, "ingest", "")
        worker.done.connect(self._on_ingested)
        worker.error.connect(lambda t: self._on_ingested({"text": t}))
        worker.finished.connect(lambda: self._drop(worker))
        self._workers.append(worker)
        worker.start()

    def _on_ingested(self, result: dict) -> None:
        self.ingest_btn.setEnabled(True)
        self.stats_label.setText(result.get("text", "Ingest complete."))
        self._refresh()

    def _drop(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
