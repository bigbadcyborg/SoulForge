"""GUI client settings — server location, auth token, and hotkeys.

Read from environment variables (set by the launcher) with sensible defaults so
the GUI works out of the box against a local WSL server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GuiSettings:
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token: str = ""
    snapshot_hotkey: str = "ctrl+alt+s"
    transcribe_hotkey: str = "ctrl+alt+a"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        url = f"ws://{self.host}:{self.port}/ws/chat"
        if self.auth_token:
            url += f"?token={self.auth_token}"
        return url

    @classmethod
    def from_env(cls) -> "GuiSettings":
        return cls(
            host=os.environ.get("SOULFORGE_HOST", "127.0.0.1"),
            port=int(os.environ.get("SOULFORGE_PORT", "8765")),
            auth_token=os.environ.get("SOULFORGE_TOKEN", ""),
            snapshot_hotkey=os.environ.get("SOULFORGE_SNAPSHOT_HOTKEY", "ctrl+alt+s"),
            transcribe_hotkey=os.environ.get("SOULFORGE_TRANSCRIBE_HOTKEY", "ctrl+alt+a"),
        )
