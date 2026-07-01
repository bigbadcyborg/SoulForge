"""JSON persistence for agent runs."""

from __future__ import annotations

import json
from pathlib import Path

from app.agents.models import AgentRun
from app.core.config import AppConfig


class AgentRunStore:
    """Stores one agent run per JSON file."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runs_dir = config.agents.runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        safe = "".join(ch for ch in run_id if ch.isalnum() or ch in ("_", "-"))
        return self.runs_dir / f"{safe}.json"

    def save(self, run: AgentRun) -> None:
        run.touch()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(run.run_id)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(run.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp.replace(path)

    def load(self, run_id: str) -> AgentRun | None:
        path = self.path_for(run_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return AgentRun.from_dict(data)

    def list_runs(self) -> list[AgentRun]:
        runs: list[AgentRun] = []
        if not self.runs_dir.exists():
            return runs
        for path in sorted(
            self.runs_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                runs.append(AgentRun.from_dict(data))
        return runs

    def latest(self) -> AgentRun | None:
        runs = self.list_runs()
        return runs[0] if runs else None

