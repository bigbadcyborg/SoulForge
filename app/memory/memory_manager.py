"""Memory manager: loads and saves durable user/project/session context files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import AppConfig

SECTION_KEYS = ("user", "memory", "session")

SECTION_FILENAMES = {
    "user": "user.md",
    "memory": "memory.md",
    "session": "session.md",
}

SECTION_TEMPLATES = {
    "user": (
        "<!-- Stable user facts: name, preferences, constraints.\n"
        "     Injected into the system prompt when the Memory feature is enabled. -->\n"
    ),
    "memory": (
        "<!-- Durable project and preference memory.\n"
        "     Injected into the system prompt when the Memory feature is enabled. -->\n"
    ),
    "session": (
        "<!-- Short-term session notes. Clear manually when a session ends.\n"
        "     Injected into the system prompt when the Memory feature is enabled. -->\n"
    ),
}


@dataclass
class MemorySnapshot:
    user: str = ""
    memory: str = ""
    session: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.user or self.memory or self.session)


class MemoryManager:
    """Reads and writes memory files, clamping them to configured character limits."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _path_and_limit(self, section: str) -> tuple[Path, int]:
        if section not in SECTION_KEYS:
            raise ValueError(
                f"Invalid memory section '{section}'. Use: {', '.join(SECTION_KEYS)}."
            )
        mem = self.config.memory
        mapping = {
            "user": (mem.user_path, mem.max_user_chars),
            "memory": (mem.memory_path, mem.max_memory_chars),
            "session": (mem.session_path, mem.max_session_chars),
        }
        return mapping[section]

    def limits(self) -> dict[str, int]:
        mem = self.config.memory
        return {
            "user": mem.max_user_chars,
            "memory": mem.max_memory_chars,
            "session": mem.max_session_chars,
        }

    def ensure_files(self) -> None:
        """Create missing memory files from starter templates."""
        for section in SECTION_KEYS:
            path, _ = self._path_and_limit(section)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(SECTION_TEMPLATES[section], encoding="utf-8")

    @staticmethod
    def _strip_html_comments(text: str) -> str:
        """Remove HTML comments so only substantive memory reaches the model."""
        return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _read_clamped(path: Path, max_chars: int) -> str:
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError as error:
            print(f"[memory] Could not read {path}: {error}")
            return ""
        text = MemoryManager._strip_html_comments(text)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return text

    def read_raw(self, section: str) -> str:
        """Return full file content for editing (no clamp)."""
        path, _ = self._path_and_limit(section)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError as error:
            print(f"[memory] Could not read {path}: {error}")
            return ""

    def save(self, section: str, content: str) -> tuple[str, bool]:
        """Write content to a memory file, clamping to the configured limit."""
        path, max_chars = self._path_and_limit(section)
        text = content.strip()
        truncated = False
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip()
            truncated = True
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text, truncated

    def load(self) -> MemorySnapshot:
        mem = self.config.memory
        return MemorySnapshot(
            user=self._read_clamped(mem.user_path, mem.max_user_chars),
            memory=self._read_clamped(mem.memory_path, mem.max_memory_chars),
            session=self._read_clamped(mem.session_path, mem.max_session_chars),
        )

    def format_view(self, snapshot: MemorySnapshot, memory_enabled: bool) -> str:
        """Format memory sections with character counts for display."""
        limits = self.limits()
        lines = [f"Memory injection: {'ON' if memory_enabled else 'OFF'}", ""]
        sections = [
            ("user", snapshot.user),
            ("memory", snapshot.memory),
            ("session", snapshot.session),
        ]
        for key, content in sections:
            filename = SECTION_FILENAMES[key]
            limit = limits[key]
            count = len(content)
            lines.append(f"--- {filename} ({count}/{limit}) ---")
            lines.append(content if content else "(empty)")
            lines.append("")
        return "\n".join(lines).rstrip()
