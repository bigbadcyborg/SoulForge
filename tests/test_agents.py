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
from app.agents.models import AgentCheckpoint, AgentRun, AgentTask, ContextPruning
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


def test_parse_tolerates_trailing_and_extra_objects() -> None:
    """Small models emit prose or a second object after the envelope."""
    envelope = _envelope(role="creator", run_id="run_1", task_id="t1")

    # 1. trailing prose after the object
    parsed = parse_agent_envelope(envelope + "\n\nHope that helps!")
    assert parsed.task_id == "t1"

    # 2. leading prose before the object
    parsed = parse_agent_envelope("Here is the envelope:\n" + envelope)
    assert parsed.task_id == "t1"

    # 3. a short fragment object emitted before the real envelope — the real
    #    one wins because it carries more envelope keys
    fragment = '{"schema_version": 1, "role": "creator"}'
    parsed = parse_agent_envelope(fragment + "\n" + envelope)
    assert parsed.task_id == "t1"
    assert parsed.status == "pass"

    # 4. fenced JSON surrounded by prose
    parsed = parse_agent_envelope(f"Sure:\n```json\n{envelope}\n```\nDone.")
    assert parsed.task_id == "t1"


def test_parse_still_rejects_non_json() -> None:
    with pytest.raises(AgentProtocolError):
        parse_agent_envelope("no json at all here")


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
            if "\"task_id\": \"final\"" in messages[-1]["content"]:
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


def test_resume_run_after_checkpoint_completes(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path, tools=True))
    runtime = MagicMock()
    runtime.warm_resident_profiles.return_value = []

    run = AgentRun(run_id="run_resume", goal="Use tool", status="paused")
    task = AgentTask(
        id="exec",
        role="executor",
        title="Exec",
        instructions="Write",
        status="paused",
        attempts=1,
    )
    run.tasks = [task]
    run.checkpoints.append(
        AgentCheckpoint(
            checkpoint_id="chk_done",
            run_id="run_resume",
            task_id="exec",
            tool_call={"name": "write_file", "args": {}},
            status="executed",
            result={"output": "wrote it"},
        )
    )

    def completion(profile, messages, stream=False):
        content = messages[-1]["content"]
        if "\"task_id\": \"final\"" in content:
            return _envelope(
                role="synthesizer",
                run_id="run_resume",
                task_id="final",
                summary="done",
            )
        if "\"task_id\": \"critic\"" in content:
            return _envelope(role="critic", run_id="run_resume", task_id="critic")
        # The resumed executor task must see its approved tool result.
        assert "wrote it" in content
        return _envelope(role="executor", run_id="run_resume", task_id="exec")

    runtime.create_chat_completion_for_profile.side_effect = completion
    manager = AgentManager(config, runtime)
    manager.active_run_id = run.run_id
    manager.store.save(run)

    result = manager.resume_run()
    assert result.success is True
    assert result.run is not None
    assert result.run.status == "completed"
    assert result.run.final_answer == "done"


def test_resume_run_requires_resolved_checkpoints(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path, tools=True))
    runtime = MagicMock()
    run = AgentRun(run_id="run_wait", goal="Use tool", status="paused")
    run.checkpoints.append(
        AgentCheckpoint(
            checkpoint_id="chk_open",
            run_id="run_wait",
            task_id="exec",
            tool_call={"name": "write_file", "args": {}},
        )
    )
    manager = AgentManager(config, runtime)
    manager.active_run_id = run.run_id
    manager.store.save(run)

    result = manager.resume_run()
    assert result.success is False
    assert "chk_open" in result.message


def test_default_roles_carry_allowed_tools(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path))
    assert config.agents.roles["researcher"].allowed_tools == [
        "read_file",
        "list_dir",
        "search_docs",
    ]
    assert "write_file" in config.agents.roles["executor"].allowed_tools


def test_allowed_tools_round_trip(tmp_path) -> None:
    from app.core.config import save_agents

    config = load_config(_write_agent_config(tmp_path))
    config_path = tmp_path / "config.yaml"
    config.agents.roles["researcher"].allowed_tools = ["read_file"]
    save_agents(config, config_path)
    reloaded = load_config(config_path)
    assert reloaded.agents.roles["researcher"].allowed_tools == ["read_file"]


def test_worker_context_includes_rag_for_researcher(tmp_path) -> None:
    from app.rag.retriever import RetrievedChunk

    config = load_config(_write_agent_config(tmp_path))
    runtime = MagicMock()

    class FakeRetriever:
        def retrieve(self, query, top_k=None):
            assert "Find facts" in query
            return [
                RetrievedChunk(
                    source="notes.md",
                    chunk_index=0,
                    distance=0.1,
                    document="The sky is blue.",
                )
            ]

    manager = AgentManager(
        config, runtime, retriever_provider=lambda: FakeRetriever()
    )
    run = AgentRun(run_id="run_ctx", goal="Learn things")
    task = AgentTask(
        id="r1", role="researcher", title="Research", instructions="Find facts"
    )
    run.tasks = [task]

    context = manager._context_for_task(run, task)
    assert "Retrieved documents (RAG)" in context
    assert "The sky is blue." in context


def test_worker_context_skips_rag_for_non_researcher(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path))
    runtime = MagicMock()
    called = []

    class FakeRetriever:
        def retrieve(self, query, top_k=None):
            called.append(query)
            return []

    manager = AgentManager(
        config, runtime, retriever_provider=lambda: FakeRetriever()
    )
    run = AgentRun(run_id="run_ctx2", goal="Build things")
    # creator does not get RAG by default and this task does not opt in.
    task = AgentTask(
        id="c1", role="creator", title="Create", instructions="Build it"
    )
    run.tasks = [task]

    manager._context_for_task(run, task)
    assert called == []


def test_allowed_tools_blocks_disallowed_tool(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path, tools=True))
    runtime = MagicMock()

    class FakeExecutor:
        def __init__(self):
            self.classified = []

        def classify(self, call):
            self.classified.append(call.name)
            raise AssertionError("classify must not run for a blocked tool")

    executor = FakeExecutor()
    manager = AgentManager(
        config, runtime, tool_executor_factory=lambda: executor
    )
    run = AgentRun(run_id="run_block", goal="x", status="running")
    # researcher's default allowed_tools excludes write_file.
    task = AgentTask(id="r1", role="researcher", title="R", instructions="find")
    run.tasks = [task]
    env = parse_agent_envelope(
        _envelope(
            role="researcher",
            run_id=run.run_id,
            task_id="r1",
            status="tool_requested",
            tool_requests=[{"name": "write_file", "args": {"path": "x", "content": "y"}}],
        )
    )
    result = env.to_result()
    paused = manager._process_tool_requests(run, task, env.tool_requests, result)
    assert paused is False
    assert run.checkpoints == []
    assert executor.classified == []
    assert any("not allowed for role 'researcher'" in err for err in result.errors)


def test_start_run_emits_progress(tmp_path) -> None:
    config = load_config(_write_agent_config(tmp_path))
    runtime = MagicMock()
    runtime.warm_resident_profiles.return_value = []

    def completion(profile, messages, stream=False):
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
                            }
                        ],
                    }
                ],
            )
        if profile == "creator":
            if "\"task_id\": \"final\"" in messages[-1]["content"]:
                return _envelope(
                    role="synthesizer",
                    run_id=manager.active_run_id,
                    task_id="final",
                    summary="final",
                )
            return _envelope(
                role="creator",
                run_id=manager.active_run_id,
                task_id="draft",
                summary="draft",
            )
        return _envelope(
            role="critic", run_id=manager.active_run_id, task_id="critic"
        )

    runtime.create_chat_completion_for_profile.side_effect = completion
    manager = AgentManager(config, runtime)
    lines: list[str] = []
    result = manager.start_run("Write something", on_progress=lines.append)
    assert result.success is True
    joined = "\n".join(lines)
    assert "planning task graph" in joined
    assert any("creator/draft" in line for line in lines)
    assert "run completed" in joined
    # Callback is cleared after the run.
    assert manager.on_progress is None
