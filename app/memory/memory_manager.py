"""Memory manager: loads durable user/project/session context files.

In Iteration 1 this only reads existing files (if any) and enforces character
limits. Files that do not exist yet simply produce empty sections, so behavior
matches the original chatbot until memory files are populated in later
iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import AppConfig


@dataclass
class MemorySnapshot:
    user: str = ""
    memory: str = ""
    session: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.user or self.memory or self.session)


class MemoryManager:
    """Reads memory files and clamps them to configured character limits."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def _read_clamped(path: Path, max_chars: int) -> str:
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError as error:
            print(f"[memory] Could not read {path}: {error}")
            return ""
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return text

    def load(self) -> MemorySnapshot:
        mem = self.config.memory
        return MemorySnapshot(
            user=self._read_clamped(mem.user_path, mem.max_user_chars),
            memory=self._read_clamped(mem.memory_path, mem.max_memory_chars),
            session=self._read_clamped(mem.session_path, mem.max_session_chars),
        )
