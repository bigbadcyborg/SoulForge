"""Tests for red-team analytics, audit logging, and episodic memory."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import (
    AppConfig,
    CuratorConfig,
    FeatureConfig,
    GenerationConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    RagConfig,
    SessionsConfig,
    SkillsConfig,
    TasksConfig,
    ToolsConfig,
)
from app.core.prompt_builder import PromptBuilder
from app.memory.episodic import (
    EpisodicMemoryResult,
    EpisodicMemoryStore,
    format_episodic_results,
)
from app.memory.memory_manager import MemorySnapshot
from app.skills.simulator import (
    SimulationPayload,
    evaluate_response,
    format_simulation_report,
    run_simulation,
)
from app.utils.audit_logger import append_audit_event


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        model=ModelConfig(
            chat_model_path=str(tmp_path / "model.gguf"),
            embedding_model_path=str(tmp_path / "embed.gguf"),
            gpu_layers=0,
        ),
        generation=GenerationConfig(max_tokens=42),
        features=FeatureConfig(rag=False, memory=True),
        rag=RagConfig(db_path=str(tmp_path / "chroma"), docs_path=str(tmp_path / "docs")),
        memory=MemoryConfig(
            user_file=str(tmp_path / "memory" / "user.md"),
            memory_file=str(tmp_path / "memory" / "memory.md"),
            session_file=str(tmp_path / "memory" / "session.md"),
        ),
        skills=SkillsConfig(
            active_path=str(tmp_path / "skills" / "active"),
            archived_path=str(tmp_path / "skills" / "archived"),
            registry_path=str(tmp_path / "skills" / "registry.json"),
        ),
        curator=CuratorConfig(),
        tasks=TasksConfig(kanban_path=str(tmp_path / "tasks" / "kanban.json")),
        sessions=SessionsConfig(store_path=str(tmp_path / "sessions")),
        logging=LoggingConfig(log_path=str(tmp_path / "logs" / "soulforge.log")),
        tools=ToolsConfig(),
    )


def test_append_audit_event_writes_jsonl(tmp_path) -> None:
    config = _config(tmp_path)
    path = tmp_path / "audit.jsonl"

    append_audit_event(
        config,
        messages=[{"role": "user", "content": "hello"}],
        params={"max_tokens": 10},
        response_text="hi",
        path=path,
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["model"] == "model.gguf"
    assert event["messages"][0]["content"] == "hello"
    assert event["response_text"] == "hi"


class FakeCollection:
    def __init__(self) -> None:
        self.upserts = []

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)

    def query(self, **kwargs):
        return {
            "ids": [["turn-1"]],
            "metadatas": [[
                {
                    "user": "What did we test?",
                    "assistant": "Prompt injection.",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "turnCount": 3,
                }
            ]],
            "distances": [[0.12]],
        }


class FakeRuntime:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_episodic_memory_store_add_and_search(tmp_path) -> None:
    store = EpisodicMemoryStore(_config(tmp_path), FakeRuntime())  # type: ignore[arg-type]
    collection = FakeCollection()
    store._get_collection = lambda: collection  # type: ignore[method-assign]

    assert store.add_turn(
        turn_id="turn-1",
        user="What did we test?",
        assistant="Prompt injection.",
        turn_count=3,
    )
    assert collection.upserts[0]["ids"] == ["turn-1"]

    results = store.search("prompt injection")
    assert results[0].assistant == "Prompt injection."
    assert results[0].distance == 0.12


def test_format_episodic_results_empty_and_nonempty() -> None:
    assert format_episodic_results([]) == "No episodic memory results."
    formatted = format_episodic_results(
        [
            EpisodicMemoryResult(
                id="abc",
                user="u",
                assistant="a",
                timestamp="now",
                turn_count=1,
            )
        ]
    )
    assert "[abc] turn 1" in formatted


def test_prompt_builder_includes_episodic_context(tmp_path) -> None:
    config = _config(tmp_path)
    prompt = PromptBuilder(config).build_user_turn(
        "What did we cover?",
        memory=MemorySnapshot(user="Russ likes green."),
        use_memory=True,
        episodic_context="Turn 1: Prompt injection boundaries.",
    )
    assert "RELEVANT EPISODIC MEMORY" in prompt
    assert "Prompt injection boundaries" in prompt


def test_simulation_evaluator_flags_forbidden_terms() -> None:
    payload = SimulationPayload(
        category="Direct prompt injection",
        name="test",
        prompt="payload",
        forbidden_terms=("Broken Access Control",),
        required_terms=("Direct prompt injection",),
    )
    passed, reason = evaluate_response(
        payload,
        "Direct prompt injection plus Broken Access Control",
    )
    assert not passed
    assert "Forbidden" in reason


class FakeSimulationRuntime:
    def create_chat_completion(self, messages, stream=False, **overrides):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Direct prompt injection, Indirect injection, "
                            "Tool abuse, Data exfiltration, Denial / cost, Persona drift"
                        )
                    }
                }
            ]
        }


def test_run_simulation_and_format_report() -> None:
    result = run_simulation(
        FakeSimulationRuntime(),  # type: ignore[arg-type]
        system_prompt="system",
        attack_type="prompt-injection",
    )
    report = format_simulation_report(result)
    assert "Simulation: prompt-injection" in report
    assert "PASS" in report
