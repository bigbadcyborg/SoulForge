"""Global (system-wide) hotkeys for the SoulForge GUI.

Hotkey callbacks fire on a background thread, so they emit Qt signals with a
queued connection to hop onto the GUI thread before touching widgets. Uses the
``keyboard`` library if available, otherwise falls back to ``pynput``.
Requires PySide6 (Windows GUI venv).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal

from gui.settings import GuiSettings
from gui.util import to_pynput_hotkey


class HotkeyManager(QObject):
    snapshot_triggered = Signal()
    transcribe_triggered = Signal()

    def __init__(self, settings: GuiSettings, window) -> None:
        super().__init__()
        self.settings = settings
        self._backend = None
        self._pynput_listener = None
        self.snapshot_triggered.connect(
            window.trigger_snapshot, Qt.QueuedConnection
        )
        self.transcribe_triggered.connect(
            window.trigger_transcribe, Qt.QueuedConnection
        )

    def start(self) -> None:
        if self._start_keyboard():
            self._backend = "keyboard"
            return
        if self._start_pynput():
            self._backend = "pynput"
            return
        raise RuntimeError(
            "No global-hotkey backend available (install 'keyboard' or 'pynput')."
        )

    def stop(self) -> None:
        if self._backend == "keyboard":
            import keyboard

            keyboard.remove_all_hotkeys()
        elif self._backend == "pynput" and self._pynput_listener is not None:
            self._pynput_listener.stop()

    # -- backends --------------------------------------------------------

    def _start_keyboard(self) -> bool:
        try:
            import keyboard
        except Exception:  # noqa: BLE001
            return False
        keyboard.add_hotkey(
            self.settings.snapshot_hotkey, self.snapshot_triggered.emit
        )
        keyboard.add_hotkey(
            self.settings.transcribe_hotkey, self.transcribe_triggered.emit
        )
        return True

    def _start_pynput(self) -> bool:
        try:
            from pynput import keyboard as pk
        except Exception:  # noqa: BLE001
            return False

        hotkeys = {
            to_pynput_hotkey(self.settings.snapshot_hotkey): self.snapshot_triggered.emit,
            to_pynput_hotkey(self.settings.transcribe_hotkey): self.transcribe_triggered.emit,
        }
        self._pynput_listener = pk.GlobalHotKeys(hotkeys)
        self._pynput_listener.start()
        return True
