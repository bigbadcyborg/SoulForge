"""Prompt templates for the SoulForge agent roles."""

from __future__ import annotations

import json

from app.agents.models import AgentRun, AgentTask

JSON_RULES = (
    "You are a SoulForge workflow agent. Return only one valid JSON object. "
    "Do not include markdown fences, commentary, or prose outside JSON. "
    "Every response must match AgentJsonEnvelope schema_version 1."
)

ENVELOPE_SHAPE = {
    "schema_version": 1,
    "role": "creator",
    "run_id": "run_id",
    "task_id": "task_id",
    "parent_task_id": None,
    "context_pruning": {
        "include_goal": True,
        "include_ancestors": True,
        "include_dependencies": True,
        "exclude_sibling_tasks": True,
        "include_rag": False,
        "include_memory": False,
        "include_skills": False,
        "max_context_chars": 6000,
    },
    "status": "pass",
    "summary": "string",
    "artifacts": [],
    "tool_requests": [],
    "next_actions": [],
    "errors": [],
}


def envelope_example(role: str, run_id: str, task_id: str) -> dict:
    """ENVELOPE_SHAPE with the role/run_id/task_id this caller must actually use.

    The generic shape hardcodes role "creator"; smaller models copy it verbatim
    and fail envelope validation ("role must be 'executor', got 'creator'").
    """
    example = dict(ENVELOPE_SHAPE)
    example["role"] = role
    example["run_id"] = run_id
    example["task_id"] = task_id
    return example


ROLE_DESCRIPTIONS: dict[str, str] = {
    "orchestrator": (
        "Break the user goal into a dependency-ordered task graph and make "
        "high-level revision decisions."
    ),
    "researcher": (
        "Gather scoped facts and context from available local context without "
        "inventing sources."
    ),
    "creator": "Build the main deliverable or draft based on task inputs.",
    "executor": (
        "Request local tools only when needed and summarize execution logs precisely."
    ),
    "critic": "Validate outputs against requirements, schemas, and success criteria.",
    "synthesizer": "Produce the final answer after the task graph has passed review.",
}


def orchestrator_messages(run: AgentRun, max_iterations: int) -> list[dict[str, str]]:
    task_schema = {
        "id": "task_short_id",
        "role": "researcher|creator|executor|critic|synthesizer",
        "title": "short title",
        "instructions": "specific task instructions",
        "dependencies": [],
        "success_criteria": "objective pass criteria",
        "parent_task_id": None,
        "context_pruning": ENVELOPE_SHAPE["context_pruning"],
        "input_spec": "manual-editable input spec",
        "max_attempts": max_iterations,
    }
    system = (
        f"{JSON_RULES}\n"
        "You are the orchestrator. Build a minimal dependency-ordered task graph. "
        "Use only these worker roles: researcher, creator, executor, critic, "
        "synthesizer. Include parent_task_id when a task refines a prior task. "
        "Do not create unrelated sibling dependencies. Use context_pruning "
        "to scope each task's context: set include_rag true for tasks that need "
        "indexed documents, include_memory true for tasks that depend on stored "
        "user/project facts, and include_skills true when a task should know "
        "which local skills exist.\n"
        "REQUIRED: 'artifacts' must contain exactly one object with "
        "\"type\": \"task_graph\" and a non-empty \"tasks\" array. An empty "
        "artifacts array is invalid and the run will fail."
    )
    user = (
        f"Run ID: {run.run_id}\n"
        f"Goal:\n{run.goal}\n\n"
        "Return one AgentJsonEnvelope for task_id 'plan'. Each task item uses "
        f"this shape:\n{json.dumps(task_schema, indent=2)}\n\n"
        "Copy this structure exactly, replacing the example tasks with tasks "
        "for the goal above:\n"
        f"{json.dumps(orchestrator_example_envelope(run, max_iterations), indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def orchestrator_example_envelope(run: AgentRun, max_iterations: int) -> dict:
    """A fully populated planner envelope to show the model what to emit.

    The generic ENVELOPE_SHAPE has an empty ``artifacts`` list, which smaller
    models copy verbatim — producing a plan with no task_graph. This example
    carries a real task_graph so the required shape is unambiguous.
    """
    def _task(task_id: str, role: str, title: str, deps: list[str]) -> dict:
        return {
            "id": task_id,
            "role": role,
            "title": title,
            "instructions": f"({title} — concrete instructions for the goal)",
            "dependencies": deps,
            "success_criteria": "Objective, checkable criteria.",
            "parent_task_id": None,
            "context_pruning": ENVELOPE_SHAPE["context_pruning"],
            "input_spec": "",
            "max_attempts": max_iterations,
        }

    return {
        "schema_version": 1,
        "role": "orchestrator",
        "run_id": run.run_id,
        "task_id": "plan",
        "parent_task_id": None,
        "context_pruning": ENVELOPE_SHAPE["context_pruning"],
        "status": "pass",
        "summary": "Task graph for the goal.",
        "artifacts": [
            {
                "type": "task_graph",
                "tasks": [
                    _task("step1", "executor", "First step", []),
                    _task("step2", "synthesizer", "Summarize the result", ["step1"]),
                ],
            }
        ],
        "tool_requests": [],
        "next_actions": [],
        "errors": [],
    }


def planner_retry_messages(
    run: AgentRun, max_iterations: int, error: str
) -> list[dict[str, str]]:
    """Re-ask the orchestrator after an unusable plan, naming what was wrong."""
    base = orchestrator_messages(run, max_iterations)
    correction = (
        f"Your previous plan was rejected: {error}\n\n"
        "Return the envelope again. The 'artifacts' array MUST contain an "
        'object {"type": "task_graph", "tasks": [ ... ]} with at least one '
        "task. Output only the JSON object."
    )
    return base + [{"role": "user", "content": correction}]


def task_messages(run: AgentRun, task: AgentTask, context: str) -> list[dict[str, str]]:
    system = (
        f"{JSON_RULES}\n"
        f"You are the {task.role} agent. {ROLE_DESCRIPTIONS.get(task.role, '')} "
        "Set status to pass, revise, blocked, or tool_requested. If you need a "
        "tool, set status to tool_requested and include tool_requests items with "
        "name, args, and rationale. Use artifacts for durable outputs."
    )
    user = (
        f"Run ID: {run.run_id}\n"
        f"Root goal:\n{run.goal}\n\n"
        f"Task ID: {task.id}\n"
        f"Parent task ID: {task.parent_task_id or 'null'}\n"
        f"Title: {task.title}\n"
        f"Instructions:\n{task.instructions}\n\n"
        f"Manual input spec:\n{task.input_spec or '(none)'}\n\n"
        f"Success criteria:\n{task.success_criteria or '(none)'}\n\n"
        f"Scoped context:\n{context or '(none)'}\n\n"
        "Return exactly this envelope shape (keep role, run_id and task_id as "
        "shown):\n"
        f"{json.dumps(envelope_example(task.role, run.run_id, task.id), indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def critic_messages(run: AgentRun, context: str) -> list[dict[str, str]]:
    system = (
        f"{JSON_RULES}\n"
        "You are the critic. Decide if the run satisfies the original goal. "
        "Use status pass only when the final deliverable is ready. Use revise "
        "with next_actions when a specific task should be corrected. Use blocked "
        "when the run cannot continue without user intervention."
    )
    user = (
        f"Run ID: {run.run_id}\n"
        f"Goal:\n{run.goal}\n\n"
        f"Run context:\n{context}\n\n"
        "Return exactly this envelope shape (keep role, run_id and task_id as "
        "shown):\n"
        f"{json.dumps(envelope_example('critic', run.run_id, 'critic'), indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def synthesizer_messages(run: AgentRun, context: str) -> list[dict[str, str]]:
    system = (
        f"{JSON_RULES}\n"
        "You are the synthesizer. Produce the final user-facing answer in the "
        "summary field and any supporting structured data in artifacts. Keep the "
        "answer direct and grounded in passed task outputs."
    )
    user = (
        f"Run ID: {run.run_id}\n"
        f"Goal:\n{run.goal}\n\n"
        f"Passed run context:\n{context}\n\n"
        "Return exactly this envelope shape (keep role, run_id and task_id as "
        "shown); put the final answer in 'summary':\n"
        f"{json.dumps(envelope_example('synthesizer', run.run_id, 'final'), indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def result_context(run: AgentRun, *, max_chars: int = 8000) -> str:
    lines = []
    for result in run.results:
        lines.append(f"[{result.role}:{result.task_id}:{result.status}]")
        if result.summary:
            lines.append(result.summary)
        if result.artifacts:
            lines.append(json.dumps(result.artifacts, ensure_ascii=False)[:2000])
        if result.errors:
            lines.append("Errors: " + "; ".join(result.errors))
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[-max_chars:]
    return text

