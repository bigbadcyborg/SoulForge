"""SoulForge desktop GUI entry point (Windows).

Run from the Windows GUI venv:  ``python -m gui.app``
Requires the WSL API server to be running (``python -m app.server`` in WSL, or
use ``start-gui-windows.ps1`` which starts both).
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QDialog

from gui.api_client import ApiClient
from gui.chat_window import ChatWindow
from gui.settings import GuiSettings
from gui.startup_dialog import StartupDialog


def main() -> None:
    settings = GuiSettings.from_env()
    app = QApplication(sys.argv)
    app.setApplicationName("SoulForge")

    # Ask which models to load before the model is (lazily) loaded, then kick
    # off the chosen loads so nothing has to load mid-session.
    startup_client = ApiClient(settings)
    dialog = StartupDialog(startup_client)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    choices = dialog.choices()
    try:
        startup_client.session_start(
            choices["chat_model"], choices["load_agents"], choices["load_vision"]
        )
    except Exception as error:  # noqa: BLE001
        print(f"[gui] session start failed: {error}")
    startup_client.close()

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
