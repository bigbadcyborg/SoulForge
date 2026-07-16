"""SoulForge desktop GUI entry point (Windows).

Run from the Windows GUI venv:  ``python -m gui.app``
Requires the WSL API server to be running (``python -m app.server`` in WSL, or
use ``start-gui-windows.ps1`` which starts both).
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui.chat_window import ChatWindow
from gui.settings import GuiSettings


def main() -> None:
    settings = GuiSettings.from_env()
    app = QApplication(sys.argv)
    app.setApplicationName("SoulForge")
    window = ChatWindow(settings)

    # Global hotkeys for snapshot/transcribe are registered here once their
    # modules land (Phases 2/3); they call window.trigger_snapshot /
    # window.trigger_transcribe.
    try:
        from gui.hotkeys import HotkeyManager

        manager = HotkeyManager(settings, window)
        manager.start()
        window._hotkey_manager = manager  # keep a reference alive
    except Exception as error:  # noqa: BLE001 - hotkeys are optional
        print(f"[gui] Global hotkeys unavailable: {error}")

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
