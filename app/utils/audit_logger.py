"""Structured JSONL audit logging for model interactions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import AppConfig, PROJECT_ROOT

AUDIT_LOG_PATH = PROJECT_ROOT / "logs" / "audit.jsonl"


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of arbitrary model data."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return repr(value)


def append_audit_event(
    config: AppConfig,
    *,
    messages: list[dict[str, str]],
    params: dict[str, Any],
    response_text: str = "",
    stream: bool = False,
    error: str = "",
    path: Path = AUDIT_LOG_PATH,
) -> None:
    """Append one model request/response event to ``logs/audit.jsonl``."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.model.chat_model.name,
        "model_path": str(config.model.chat_model),
        "chat_format": config.model.chat_format,
        "stream": stream,
        "generation_params": _json_safe(params),
        "messages": _json_safe(messages),
        "response_text": response_text,
        "error": error,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Audit logging should never break chat, but failures should be visible.
        print(f"[audit] Failed to write audit event: {exc}")
