"""Data models for SoulForge multi-agent runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.tools.models import ToolRisk

AGENT_ROLES: tuple[str, ...] = (
    "orchestrator",
    "researcher",
    "creator",
    "executor",
    "critic",
    "synthesizer",
)

ENVELOPE_STATUSES: tuple[str, ...] = (
    "pass",
    "revise",
    "blocked",
    "tool_requested",
)

TASK_STATUSES: tuple[str, ...] = (
    "pending",
    "running",
    "passed",
    "blocked",
    "paused",
    "revising",
)

RUN_STATUSES: tuple[str, ...] = (
    "created",
    "running",
    "paused",
    "blocked",
    "completed",
    "cancelled",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class ContextPruning:
    include_goal: bool = True
    include_ancestors: bool = True
    include_dependencies: bool = True
    exclude_sibling_tasks: bool = True
    max_context_chars: int = 6000

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_goal": self.include_goal,
            "include_ancestors": self.include_ancestors,
            "include_dependencies": self.include_dependencies,
            "exclude_sibling_tasks": self.exclude_sibling_tasks,
            "max_context_chars": self.max_context_chars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ContextPruning:
        data = data if isinstance(data, dict) else {}
        return cls(
            include_goal=bool(data.get("include_goal", True)),
            include_ancestors=bool(data.get("include_ancestors", True)),
            include_dependencies=bool(data.get("include_dependencies", True)),
            exclude_sibling_tasks=bool(data.get("exclude_sibling_tasks", True)),
            max_context_chars=int(data.get("max_context_chars", 6000) or 6000),
        )


@dataclass
class AgentRole:
    name: str
    description: str
    model_profile: str
    allowed_tools: list[str] = field(default_factory=list)
    system_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model_profile": self.model_profile,
            "allowed_tools": list(self.allowed_tools),
            "system_prompt": self.system_prompt,
        }


@dataclass
class AgentTask:
    id: str
    role: str
    title: str
    instructions: str
    dependencies: list[str] = field(default_factory=list)
    success_criteria: str = ""
    parent_task_id: str | None = None
    context_pruning: ContextPruning = field(default_factory=ContextPruning)
    input_spec: str = ""
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 1
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "title": self.title,
            "instructions": self.instructions,
            "dependencies": list(self.dependencies),
            "success_criteria": self.success_criteria,
            "parent_task_id": self.parent_task_id,
            "context_pruning": self.context_pruning.to_dict(),
            "input_spec": self.input_spec,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTask:
        return cls(
            id=str(data.get("id", "")),
            role=str(data.get("role", "")),
            title=str(data.get("title", "")),
            instructions=str(data.get("instructions", "")),
            dependencies=[str(item) for item in data.get("dependencies", []) if item],
            success_criteria=str(data.get("success_criteria", "")),
            parent_task_id=(
                str(data["parent_task_id"])
                if data.get("parent_task_id") not in (None, "")
                else None
            ),
            context_pruning=ContextPruning.from_dict(data.get("context_pruning")),
            input_spec=str(data.get("input_spec", "")),
            status=str(data.get("status", "pending")),
            attempts=int(data.get("attempts", 0) or 0),
            max_attempts=int(data.get("max_attempts", 1) or 1),
            updated_at=str(data.get("updated_at", utc_now())),
        )


@dataclass
class AgentResult:
    task_id: str
    role: str
    status: str
    summary: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "status": self.status,
            "summary": self.summary,
            "artifacts": list(self.artifacts),
            "tool_calls": list(self.tool_calls),
            "errors": list(self.errors),
            "raw": self.raw,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResult:
        return cls(
            task_id=str(data.get("task_id", "")),
            role=str(data.get("role", "")),
            status=str(data.get("status", "")),
            summary=str(data.get("summary", "")),
            artifacts=list(data.get("artifacts", []) or []),
            tool_calls=list(data.get("tool_calls", []) or []),
            errors=[str(item) for item in data.get("errors", [])],
            raw=str(data.get("raw", "")),
            created_at=str(data.get("created_at", utc_now())),
        )


@dataclass
class AgentCheckpoint:
    checkpoint_id: str
    run_id: str
    task_id: str
    tool_call: dict[str, Any]
    risk: str = ToolRisk.ACTION.value
    status: str = "pending"  # pending | approved | rejected | executed | failed
    result: dict[str, Any] | None = None
    created_at: str = field(default_factory=utc_now)
    resolved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "tool_call": dict(self.tool_call),
            "risk": self.risk,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCheckpoint:
        return cls(
            checkpoint_id=str(data.get("checkpoint_id", "")),
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            tool_call=dict(data.get("tool_call", {}) or {}),
            risk=str(data.get("risk", ToolRisk.ACTION.value)),
            status=str(data.get("status", "pending")),
            result=data.get("result"),
            created_at=str(data.get("created_at", utc_now())),
            resolved_at=str(data.get("resolved_at", "")),
        )


@dataclass
class AgentJsonEnvelope:
    schema_version: int
    role: str
    run_id: str
    task_id: str
    parent_task_id: str | None
    context_pruning: ContextPruning
    status: str
    summary: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    tool_requests: list[dict[str, Any]] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_result(self, raw: str = "") -> AgentResult:
        return AgentResult(
            task_id=self.task_id,
            role=self.role,
            status=self.status,
            summary=self.summary,
            artifacts=self.artifacts,
            tool_calls=self.tool_requests,
            errors=self.errors,
            raw=raw,
        )


@dataclass
class AgentRun:
    run_id: str
    goal: str
    status: str = "created"
    tasks: list[AgentTask] = field(default_factory=list)
    results: list[AgentResult] = field(default_factory=list)
    checkpoints: list[AgentCheckpoint] = field(default_factory=list)
    final_answer: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status,
            "tasks": [task.to_dict() for task in self.tasks],
            "results": [result.to_dict() for result in self.results],
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "final_answer": self.final_answer,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRun:
        return cls(
            run_id=str(data.get("run_id", "")),
            goal=str(data.get("goal", "")),
            status=str(data.get("status", "created")),
            tasks=[
                AgentTask.from_dict(item)
                for item in data.get("tasks", [])
                if isinstance(item, dict)
            ],
            results=[
                AgentResult.from_dict(item)
                for item in data.get("results", [])
                if isinstance(item, dict)
            ],
            checkpoints=[
                AgentCheckpoint.from_dict(item)
                for item in data.get("checkpoints", [])
                if isinstance(item, dict)
            ],
            final_answer=str(data.get("final_answer", "")),
            created_at=str(data.get("created_at", utc_now())),
            updated_at=str(data.get("updated_at", utc_now())),
        )

    def touch(self) -> None:
        self.updated_at = utc_now()

