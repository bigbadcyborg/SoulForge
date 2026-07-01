"""Tests for multi-agent orchestration primitives."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from app.agents.json_protocol import (
    AgentProtocolError,
    dependency_order,
    parse_agent_envelope,
    tasks_from_planner_envelope,
)
from app.agents.manager import AgentManager
from app.agents.models import AgentRun, AgentTask, ContextPruning
from app.agents.store import AgentRunStore
from app.core.chat_controller import ChatController
from app.core.config import load_config
from app.tools.models import ToolResult, ToolRisk


def _write_agent_config(tmp_path: Path, *, tools: bool = False) -> Path:
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    chat = models / "chat.gguf"
    creator = models / "creator.gguf"
    critic = models / "critic.gguf"
    chat.write_bytes(b"chat")
    creator.write_bytes(b"creator")
    critic.write_bytes(b"critic")
    data = {
        "model": {
            "chatModelPath": str(chat),
            "embeddingModelPath": str(models / "embed.gguf"),
            "gpuLayers": 0,
        },
        "features": {
            "agents": True,
            "tools": tools,
            "rag": False,
            "memory": False,
            "skills": False,
            "kanban": False,
        },
        "agents": {
            "runsPath": str(tmp_path / "agents" / "runs"),
            "maxIterations": 2,
            "strictJson": True,
            "modelProfiles": {
                "orchestrator": {"chatModelPath": str(chat), "residency": "swap"},
                "creator": {"chatModelPath": str(creator), "residency": "resident"},
                "critic_executor": {"chatModelPath": str(critic), "residency": "resident"},
            },
        },
        "rag": {"dbPath": str(tmp_path / "chroma"), "docsPath": str(tmp_path / "docs")},
        "memory": {
            "userFile": str(tmp_path / "memory" / "user.md"),
            "memoryFile": str(tmp_path / "memory" / "memory.md"),
            "sessionFile": str(tmp_path / "memory" / "session.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "skills" / "active"),
            "archivedPath": str(tmp_path / "skills" / "archived"),
            "registryPath": str(tmp_path / "skills" / "registry.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "tasks" / "kanban.json")},
        "sessions": {"storePath": str(tmp_path / "sessions")},
        "tools": {
            "allowWrite": True,
            "allowShell": False,
            "readRoots": [str(tmp_path)],
            "writeRoots": [str(tmp_path)],
            "autoApproveReadOnly": True,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _envelope(
    *,
    role: str,
    run_id: str,
    task_id: str,
    status: str = "pass",
    summary: str = "ok",
    artifacts=None,
    next_actions=None,
    tool_requests=None,
) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "role": role,
            "run_id": run_id,
            "task_id": task_id,
            "parent_task_id": None,
            "context_pruning": ContextPruning().to_dict(),
            "status": status,
            "summary": summary,
            "artifacts": artifacts or [],
            "tool_requests": tool_requests or [],
            "next_actions": next_actions or [],
            "errors": [],
        }
    )


def test_parse_agent_envelope_requires_strict_shape() -> None:
    with pytest.raises(AgentProtocolError):
        parse_agent_envelope('{"role": "creator"}')
    raw = _envelope(role="creator", run_id="run_1", task_id="task_1")
    parsed = parse_agent_envelope(raw, expected_run_id="run_1", expected_role="creator")
    assert parsed.context_pruning.exclude_sibling_tasks is True
    assert parsed.parent_task_id is None


def test_task_graph_validation_and_dependency_order() -> None:
    plan = parse_agent_envelope(
        _envelope(
            role="orchestrator",
            run_id="run_1",
            task_id="plan",
            artifacts=[
                {
                    "type": "task_graph",
                    "tasks": [
                        {
                            "id": "create",
                            "role": "creator",
                            "title": "Create",
                            "instructions": "Build it",
                            "dependencies": ["research"],
                            "parent_task_id": "research",
                        },
                        {
                            "id": "research",
                            "role": "researcher",
                            "title": "Research",
                            "instructions": "Find facts",
                        },
                    ],
                }
            ],
        )
    )
    tasks = tasks_from_planner_envelope(plan, default_max_attempts=2)
    assert [task.id for task in dependency_order(tasks)] == ["research", "create"]
    assert tasks[0].parent_task_id == "research"


def test_agent_run_store_roundtrip(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path))
    store = AgentRunStore(config)
    run = AgentRun(run_id="run_test", goal="Do thing")
    run.tasks.append(
        AgentTask(id="t1", role="creator", title="T", instructions="Do")
    )
    store.save(run)
    loaded = store.load("run_test")
    assert loaded is not None
    assert loaded.tasks[0].id == "t1"


def test_agent_manager_repairs_bad_json_and_completes(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path))
    runtime = MagicMock()
    responses = []

    def completion(profile, messages, stream=False):
        if profile == "orchestrator" and len(responses) == 0:
            responses.append("bad")
            return "not-json"
        if profile == "orchestrator":
            return _envelope(
                role="orchestrator",
                run_id=manager.active_run_id,
                task_id="plan",
                artifacts=[
                    {
                        "type": "task_graph",
                        "tasks": [
                            {
                                "id": "draft",
                                "role": "creator",
                                "title": "Draft",
                                "instructions": "Draft answer",
                                "input_spec": "initial",
                            }
                        ],
                    }
                ],
            )
        if profile == "creator":
            if "task_id 'final'" in messages[-1]["content"]:
                return _envelope(
                    role="synthesizer",
                    run_id=manager.active_run_id,
                    task_id="final",
                    summary="final answer",
                )
            return _envelope(
                role="creator",
                run_id=manager.active_run_id,
                task_id="draft",
                summary="draft done",
            )
        return _envelope(
            role="critic",
            run_id=manager.active_run_id,
            task_id="critic",
            summary="passed",
        )

    runtime.create_chat_completion_for_profile.side_effect = completion
    runtime.warm_resident_profiles.return_value = []
    manager = AgentManager(config, runtime)
    result = manager.start_run("Write something")
    assert result.success is True
    assert result.run is not None
    assert result.run.status == "completed"
    assert result.run.final_answer == "final answer"


def test_agent_checkpoint_requires_approval(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path, tools=True))
    runtime = MagicMock()
    runtime.warm_resident_profiles.return_value = []

    class FakeExecutor:
        def classify(self, call):
            return MagicMock(
                call_id="chk1",
                call=call,
                risk=ToolRisk.WRITE,
                requires_approval=True,
            )

        def execute(self, pending):
            return ToolResult(
                call_id=pending.call_id,
                name=pending.call.name,
                success=True,
                output="wrote",
            )

    runtime.create_chat_completion_for_profile.side_effect = [
        _envelope(
            role="orchestrator",
            run_id="run_x",
            task_id="plan",
        )
    ]
    manager = AgentManager(config, runtime, tool_executor_factory=lambda: FakeExecutor())
    run = AgentRun(run_id="run_manual", goal="Use tool", status="running")
    task = AgentTask(id="exec", role="executor", title="Exec", instructions="Write")
    run.tasks = [task]
    manager.active_run_id = run.run_id
    manager.store.save(run)
    env = parse_agent_envelope(
        _envelope(
            role="executor",
            run_id=run.run_id,
            task_id="exec",
            status="tool_requested",
            tool_requests=[
                {
                    "name": "write_file",
                    "args": {"path": "x.txt", "content": "x"},
                    "rationale": "test",
                }
            ],
        )
    )
    result = env.to_result()
    assert manager._process_tool_requests(run, task, env.tool_requests, result) is True
    manager.store.save(run)
    approved = manager.approve_checkpoint("chk1")
    assert approved.success is True
