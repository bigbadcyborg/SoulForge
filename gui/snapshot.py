"""Screen-region snapshot capture for the GUI (Windows).

The selector freezes the screen into a QPixmap first, paints that frozen image
(dimmed) as a full-screen overlay, and lets the user drag a rectangle over it.
The selected region is cropped from the frozen pixmap and emitted as PNG bytes.
Freezing first avoids the translucent-overlay black-screen problems and ensures
the overlay itself can never appear in the capture. Requires PySide6 (Windows
GUI venv). The pure rectangle math in :func:`normalize_rect` stays importable
for tests.
"""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QIODevice, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from gui.api_client import ApiClient
from gui.streaming import QThread
from gui.util import capture_region_png, normalize_rect  # noqa: F401 (re-export)


def _pixmap_to_png(pixmap: QPixmap) -> bytes:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(buffer.data())


class RegionSelector(QWidget):
    """Full-screen frozen-screenshot overlay; emits the selected region as PNG."""

    region_selected = Signal(bytes)  # cropped PNG bytes
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        screen = QGuiApplication.primaryScreen()
        self._pixmap = screen.grabWindow(0)
        self._geometry = screen.geometry()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setGeometry(self._geometry)
        self.setCursor(Qt.CrossCursor)
        self._origin = None
        self._current = None

    def _local_point(self, event):
        return event.position().toPoint()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._origin = self._local_point(event)
        self._current = self._origin
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._current = self._local_point(event)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._origin is None:
            self.cancelled.emit()
            self.close()
            return
        rect = QRect(self._origin, self._local_point(event)).normalized()
        self.close()
        if rect.width() < 4 or rect.height() < 4:
            self.cancelled.emit()
            return
        # Map logical widget coords to device pixels for the cropped image.
        dpr = self._pixmap.devicePixelRatio()
        src = QRect(
            int(rect.x() * dpr),
            int(rect.y() * dpr),
            int(rect.width() * dpr),
            int(rect.height() * dpr),
        )
        cropped = self._pixmap.copy(src)
        self.region_selected.emit(_pixmap_to_png(cropped))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        # Frozen screenshot as the background, then a dim layer over everything.
        painter.drawPixmap(self.rect(), self._pixmap)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
        if self._origin and self._current:
            rect = QRect(self._origin, self._current).normalized()
            dpr = self._pixmap.devicePixelRatio()
            src = QRect(
                int(rect.x() * dpr),
                int(rect.y() * dpr),
                int(rect.width() * dpr),
                int(rect.height() * dpr),
            )
            # Redraw the selected region at full brightness + a border.
            painter.drawPixmap(rect, self._pixmap, src)
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
