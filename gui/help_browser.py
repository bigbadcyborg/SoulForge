"""Help browser dialog.

A user-friendly way to navigate every /help option: a searchable category tree
on the left, a detail pane on the right. Clicking a command inserts its usage
into the chat input; the detailed guides show their full text. Requires PySide6
(Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.api_client import ApiClient
from gui.util import command_matches

# QTreeWidgetItem roles
_ROLE_KIND = Qt.ItemDataRole.UserRole
_ROLE_USAGE = Qt.ItemDataRole.UserRole + 1
_ROLE_DESC = Qt.ItemDataRole.UserRole + 2
_ROLE_GUIDE = Qt.ItemDataRole.UserRole + 3


class HelpBrowserDialog(QDialog):
    """Browse/search all commands and open the detailed guides."""

    insert_requested = Signal(str)  # command usage to drop into the chat input

    def __init__(self, client: ApiClient, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Help")
        self.resize(820, 600)
        self._catalog = self._load_catalog()
        self._build_ui()
        self._populate()

    def _load_catalog(self) -> dict:
        try:
            return self.client.command("help", "catalog").get("data", {})
        except Exception:  # noqa: BLE001
            return {}

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        body = QHBoxLayout()

        # Left: search + tree
        left = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search commands…")
        self.search.textChanged.connect(self._apply_filter)
        left.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._on_selection)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        left.addWidget(self.tree, stretch=1)
        left_w = QWidget()
        left_w.setLayout(left)
        body.addWidget(left_w, stretch=2)

        # Right: detail pane
        right = QVBoxLayout()
        right.addWidget(QLabel("Details"))
        self.detail = QPlainTextEdit(readOnly=True)
        right.addWidget(self.detail, stretch=1)
        self.insert_btn = QPushButton("Insert into input")
        self.insert_btn.setEnabled(False)
        self.insert_btn.clicked.connect(self._insert_current)
        right.addWidget(self.insert_btn)
        right_w = QWidget()
        right_w.setLayout(right)
        body.addWidget(right_w, stretch=3)

        outer.addLayout(body, stretch=1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        outer.addWidget(close_btn)

    def _populate(self) -> None:
        self.tree.clear()
        for category in self._catalog.get("categories", []):
            parent = QTreeWidgetItem(self.tree, [category["name"]])
            parent.setData(0, _ROLE_KIND, "category")
            parent.setExpanded(True)
            for cmd in category["commands"]:
                child = QTreeWidgetItem(parent, [cmd["usage"]])
                child.setData(0, _ROLE_KIND, "command")
                child.setData(0, _ROLE_USAGE, cmd["usage"])
                child.setData(0, _ROLE_DESC, cmd["description"])
        guides = self._catalog.get("guides", [])
        if guides:
            guides_parent = QTreeWidgetItem(self.tree, ["Guides"])
            guides_parent.setData(0, _ROLE_KIND, "category")
            guides_parent.setExpanded(True)
            for guide in guides:
                child = QTreeWidgetItem(guides_parent, [guide["title"]])
                child.setData(0, _ROLE_KIND, "guide")
                child.setData(0, _ROLE_GUIDE, guide["key"])

    # -- interaction -----------------------------------------------------

    def _apply_filter(self, query: str) -> None:
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            any_visible = False
            for j in range(parent.childCount()):
                child = parent.child(j)
                kind = child.data(0, _ROLE_KIND)
                if kind == "command":
                    match = command_matches(
                        query,
                        child.data(0, _ROLE_USAGE) or "",
                        child.data(0, _ROLE_DESC) or "",
                    )
                else:  # guide: match on its title
                    match = command_matches(query, child.text(0), "")
                child.setHidden(not match)
                any_visible = any_visible or match
            parent.setHidden(not any_visible)
            if any_visible and query.strip():
                parent.setExpanded(True)

    def _on_selection(self, current, _previous) -> None:
        if current is None:
            self.insert_btn.setEnabled(False)
            return
        kind = current.data(0, _ROLE_KIND)
        if kind == "command":
            usage = current.data(0, _ROLE_USAGE) or ""
            desc = current.data(0, _ROLE_DESC) or ""
            self.detail.setPlainText(f"{usage}\n\n{desc}")
            self.insert_btn.setEnabled(True)
        elif kind == "guide":
            self.insert_btn.setEnabled(False)
            key = current.data(0, _ROLE_GUIDE)
            self.detail.setPlainText("Loading…")
            try:
                text = self.client.command("help", key).get("text", "")
            except Exception as error:  # noqa: BLE001
                text = f"Could not load guide: {error}"
            self.detail.setPlainText(text)
        else:
            self.detail.setPlainText("")
            self.insert_btn.setEnabled(False)

    def _on_double_click(self, item, _column) -> None:
        if item is not None and item.data(0, _ROLE_KIND) == "command":
            self.insert_requested.emit(item.data(0, _ROLE_USAGE) or "")

    def _insert_current(self) -> None:
        item = self.tree.currentItem()
        if item is not None and item.data(0, _ROLE_KIND) == "command":
            self.insert_requested.emit(item.data(0, _ROLE_USAGE) or "")
