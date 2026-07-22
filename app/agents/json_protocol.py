"""Strict JSON protocol parsing and validation for agent messages."""

from __future__ import annotations

import json
import re
from typing import Any

from app.agents.models import (
    AGENT_ROLES,
    ENVELOPE_STATUSES,
    AgentJsonEnvelope,
    AgentTask,
    ContextPruning,
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


class AgentProtocolError(ValueError):
    """Raised when an agent response does not match the required JSON shape."""


def _raw_json_text(text: str) -> str:
    stripped = text.strip()
    match = _JSON_BLOCK_RE.fullmatch(stripped)
    if match:
        return match.group(1).strip()
    # A fenced block may also sit inside surrounding prose.
    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


# Keys that mark a dict as an agent envelope rather than an incidental object.
_ENVELOPE_KEYS = frozenset(
    {"schema_version", "role", "run_id", "task_id", "status", "summary", "artifacts"}
)


def _iter_json_objects(text: str):
    """Yield every top-level JSON object found in *text*, in order.

    Small models often emit an object followed by prose, or split the envelope
    across several objects. ``json.loads`` rejects all of that ("Extra data"),
    so scan instead of demanding the whole string be one value.
    """
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        start = text.find("{", index)
        if start == -1:
            return
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1  # not a valid object here; keep scanning
            continue
        if isinstance(obj, dict):
            yield obj
        index = end


def _unwrap_envelope(obj: dict[str, Any]) -> dict[str, Any]:
    """Return the envelope itself when a model nests it under a wrapper key.

    Models sometimes reply {"envelope": {...}} or {"AgentJsonEnvelope": {...}};
    prefer an inner object that looks more like the envelope than the outer one.
    """
    outer_score = len(_ENVELOPE_KEYS & obj.keys())
    if outer_score >= 3:
        return obj
    for value in obj.values():
        if isinstance(value, dict) and len(_ENVELOPE_KEYS & value.keys()) > outer_score:
            return value
    return obj


def parse_json_object(text: str) -> dict[str, Any]:
    raw = _raw_json_text(text)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return _unwrap_envelope(data)
        raise AgentProtocolError("Agent response must be a JSON object.")
    except json.JSONDecodeError as error:
        # Tolerate extra data / surrounding prose: pick the object that looks
        # most like an envelope (most envelope keys, ties broken by size).
        candidates = list(_iter_json_objects(raw))
        if not candidates:
            raise AgentProtocolError(f"Invalid JSON: {error}") from error
        best = max(
            candidates,
            key=lambda obj: (len(_ENVELOPE_KEYS & obj.keys()), len(obj)),
        )
        return _unwrap_envelope(best)


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentProtocolError(f"{field_name} must be a list.")
    return [str(item) for item in value]


def _int_value(value: Any, default: int, field_name: str) -> int:
    """Coerce a model-supplied value to int, raising AgentProtocolError on junk.

    A plain ValueError here would bypass the JSON repair path, which only
    catches AgentProtocolError.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise AgentProtocolError(f"{field_name} must be an integer.") from error


def _dict_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentProtocolError(f"{field_name} must be a list.")
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise AgentProtocolError(f"{field_name} items must be JSON objects.")
        items.append(item)
    return items


def parse_agent_envelope(
    text: str,
    *,
    expected_run_id: str | None = None,
    expected_task_id: str | None = None,
    expected_role: str | None = None,
) -> AgentJsonEnvelope:
    data = parse_json_object(text)

    # Models frequently drop this bookkeeping field. Absent means "current"
    # rather than invalid; only an explicitly different version is an error.
    schema_version = _int_value(data.get("schema_version"), 1, "schema_version")
    if schema_version != 1:
        raise AgentProtocolError(
            f"schema_version must be 1, got {schema_version}."
        )

    # role/run_id/task_id are bookkeeping the caller already knows. When the
    # model omits one, filling it in beats discarding an otherwise good plan;
    # only a value that contradicts the expectation is a real error.
    role = str(data.get("role", "")).strip()
    if not role and expected_role:
        role = expected_role
    if role not in AGENT_ROLES:
        raise AgentProtocolError(
            f"role must be one of: {', '.join(AGENT_ROLES)}; got '{role}'."
        )
    if expected_role and role != expected_role:
        raise AgentProtocolError(f"role must be '{expected_role}', got '{role}'.")

    run_id = str(data.get("run_id", "")).strip() or (expected_run_id or "")
    task_id = str(data.get("task_id", "")).strip() or (expected_task_id or "")
    if not run_id:
        raise AgentProtocolError("run_id is required.")
    if not task_id:
        raise AgentProtocolError("task_id is required.")
    if expected_run_id and run_id != expected_run_id:
        raise AgentProtocolError(f"run_id must be '{expected_run_id}', got '{run_id}'.")
    if expected_task_id and task_id != expected_task_id:
        raise AgentProtocolError(
            f"task_id must be '{expected_task_id}', got '{task_id}'."
        )

    status = str(data.get("status", "")).strip()
    if status not in ENVELOPE_STATUSES:
        raise AgentProtocolError(
            f"status must be one of: {', '.join(ENVELOPE_STATUSES)}."
        )

    parent_task_id = data.get("parent_task_id")
    if parent_task_id in ("", None):
        parent_task_id = None
    else:
        parent_task_id = str(parent_task_id)

    return AgentJsonEnvelope(
        schema_version=schema_version,
        role=role,
        run_id=run_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        context_pruning=ContextPruning.from_dict(data.get("context_pruning")),
        status=status,
        summary=str(data.get("summary", "")),
        artifacts=_dict_list(data.get("artifacts", []), "artifacts"),
        tool_requests=_dict_list(data.get("tool_requests", []), "tool_requests"),
        next_actions=_dict_list(data.get("next_actions", []), "next_actions"),
        errors=_string_list(data.get("errors", []), "errors"),
    )


def task_from_json(data: dict[str, Any], *, default_max_attempts: int) -> AgentTask:
    task_id = str(data.get("id", "")).strip()
    if not task_id:
        raise AgentProtocolError("Task id is required.")

    role = str(data.get("role", "")).strip()
    if role not in AGENT_ROLES or role == "orchestrator":
        raise AgentProtocolError(
            "Task role must be a worker role: researcher, creator, executor, critic, or synthesizer."
        )

    title = str(data.get("title", "")).strip()
    instructions = str(data.get("instructions", "")).strip()
    if not title:
        raise AgentProtocolError(f"Task {task_id} title is required.")
    if not instructions:
        raise AgentProtocolError(f"Task {task_id} instructions are required.")

    parent_task_id = data.get("parent_task_id")
    if parent_task_id in ("", None):
        parent_task_id = None
    else:
        parent_task_id = str(parent_task_id)

    return AgentTask(
        id=task_id,
        role=role,
        title=title,
        instructions=instructions,
        dependencies=_string_list(data.get("dependencies", []), "dependencies"),
        success_criteria=str(data.get("success_criteria", "")),
        parent_task_id=parent_task_id,
        context_pruning=ContextPruning.from_dict(data.get("context_pruning")),
        input_spec=str(data.get("input_spec", "")),
        max_attempts=_int_value(data.get("max_attempts"), default_max_attempts, "max_attempts")
        or default_max_attempts,
    )


def tasks_from_planner_envelope(
    envelope: AgentJsonEnvelope,
    *,
    default_max_attempts: int,
) -> list[AgentTask]:
    task_items: list[dict[str, Any]] = []
    for artifact in envelope.artifacts:
        if artifact.get("type") == "task_graph" and isinstance(artifact.get("tasks"), list):
            task_items.extend(
                item for item in artifact["tasks"] if isinstance(item, dict)
            )
    if not task_items:
        for action in envelope.next_actions:
            if action.get("type") == "task" or "role" in action:
                task_items.append(action)
    if not task_items:
        raise AgentProtocolError(
            "Orchestrator response must include a task_graph artifact with tasks."
        )

    tasks = [
        task_from_json(item, default_max_attempts=default_max_attempts)
        for item in task_items
    ]
    validate_task_graph(tasks)
    return tasks


def validate_task_graph(tasks: list[AgentTask]) -> None:
    if not tasks:
        raise AgentProtocolError("Task graph cannot be empty.")
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise AgentProtocolError("Task ids must be unique.")
    known = set(ids)
    for task in tasks:
        unknown = [dep for dep in task.dependencies if dep not in known]
        if unknown:
            raise AgentProtocolError(
                f"Task {task.id} depends on unknown task(s): {', '.join(unknown)}."
            )
        if task.parent_task_id and task.parent_task_id not in known:
            raise AgentProtocolError(
                f"Task {task.id} has unknown parent_task_id: {task.parent_task_id}."
            )

    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {task.id: task for task in tasks}

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise AgentProtocolError("Task graph contains a dependency cycle.")
        visiting.add(task_id)
        for dep in by_id[task_id].dependencies:
            visit(dep)
        visiting.remove(task_id)
        visited.add(task_id)

    for task in tasks:
        visit(task.id)


def dependency_order(tasks: list[AgentTask]) -> list[AgentTask]:
    validate_task_graph(tasks)
    ordered: list[AgentTask] = []
    seen: set[str] = set()
    by_id = {task.id: task for task in tasks}

    def add(task: AgentTask) -> None:
        if task.id in seen:
            return
        for dep in task.dependencies:
            add(by_id[dep])
        seen.add(task.id)
        ordered.append(task)

    for task in tasks:
        add(task)
    return ordered


def repair_prompt(raw_text: str, error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return only one valid JSON object matching the SoulForge "
                "AgentJsonEnvelope schema. Do not include markdown or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The previous response failed validation:\n{error}\n\n"
                f"Previous response:\n{raw_text}\n\n"
                "Repair it now as strict JSON."
            ),
        },
    ]

