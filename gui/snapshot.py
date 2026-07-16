"""Screen-region snapshot capture for the GUI (Windows).

A translucent full-screen overlay lets the user drag a rectangle; the region is
grabbed with ``mss`` into PNG bytes and uploaded to the WSL vision endpoint. The
rectangle math (:func:`normalize_rect`) is a pure function kept importable for
tests; the widget and ``mss`` capture require PySide6/mss (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from gui.api_client import ApiClient
from gui.streaming import QThread
from gui.util import capture_region_png, normalize_rect  # noqa: F401 (re-export)


class RegionSelector(QWidget):
    """Full-screen translucent overlay; emits the chosen region in screen px."""

    region_selected = Signal(int, int, int, int)  # left, top, width, height
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setWindowState(Qt.WindowFullScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self._origin = None
        self._current = None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._origin = event.globalPosition().toPoint()
        self._current = self._origin
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._current = event.globalPosition().toPoint()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._origin is None:
            self.cancelled.emit()
            self.close()
            return
        end = event.globalPosition().toPoint()
        left, top, width, height = normalize_rect(
            self._origin.x(), self._origin.y(), end.x(), end.y()
        )
        self.close()
        if width < 4 or height < 4:
            self.cancelled.emit()
        else:
            self.region_selected.emit(left, top, width, height)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 70))
        if self._origin and self._current:
            rect = QRect(self._origin, self._current).normalized()
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            painter.setPen(QColor(80, 180, 255))
            painter.drawRect(rect)


class SnapshotWorker(QThread):
    """Uploads a captured PNG to the vision endpoint off the UI thread."""

    done = Signal(str)
    error = Signal(str)

    def __init__(self, client: ApiClient, image_png: bytes, prompt: str) -> None:
        super().__init__()
        self._client = client
        self._png = image_png
        self._prompt = prompt

    def run(self) -> None:
        try:
            result = self._client.snapshot(self._png, self._prompt)
            self.done.emit(result.get("text", "(no answer)"))
        except Exception as error:  # noqa: BLE001
            self.error.emit(f"Snapshot failed: {error}")
