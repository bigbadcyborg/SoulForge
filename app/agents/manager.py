"""Agent orchestration manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.agents.json_protocol import (
    AgentProtocolError,
    dependency_order,
    parse_agent_envelope,
    repair_prompt,
    tasks_from_planner_envelope,
)
from app.agents.models import (
    AgentCheckpoint,
    AgentRun,
    AgentTask,
    ContextPruning,
    new_id,
    utc_now,
)
from app.agents.prompts import (
    critic_messages,
    orchestrator_messages,
    result_context,
    synthesizer_messages,
    task_messages,
)
from app.agents.store import AgentRunStore
from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime
from app.tools.executor import ToolExecutor
from app.tools.models import PendingToolCall, ToolCall, ToolResult
from app.tools.permissions import tool_risk

ToolExecutorFactory = Callable[[], ToolExecutor]


@dataclass
class AgentActionResult:
    success: bool
    message: str
    run: AgentRun | None = None


class AgentManager:
    """Coordinates local model profiles, agent prompts, and run persistence."""

    def __init__(
        self,
        config: AppConfig,
        runtime: ModelRuntime,
        *,
        tool_executor_factory: ToolExecutorFactory | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.store = AgentRunStore(config)
        self.tool_executor_factory = tool_executor_factory
        self.active_run_id: str = ""

    def start_run(self, goal: str) -> AgentActionResult:
        goal = goal.strip()
        if not goal:
            return AgentActionResult(False, "Usage: /agents run <goal>")

        run = AgentRun(run_id=new_id("run"), goal=goal, status="running")
        self.active_run_id = run.run_id
        self.store.save(run)

        warnings = self.runtime.warm_resident_profiles()
        try:
            plan = self._run_envelope(
                role="orchestrator",
                task_id="plan",
                run=run,
                messages=orchestrator_messages(
                    run,
                    self.config.agents.max_iterations,
                ),
            )
            run.results.append(plan.to_result())
            run.tasks = tasks_from_planner_envelope(
                plan,
                default_max_attempts=self.config.agents.max_iterations,
            )
            self.store.save(run)
            self._execute_run(run)
        except Exception as error:  # noqa: BLE001
            run.status = "blocked"
            run.results.append(self._error_result(str(error)))
            self.store.save(run)
            warning_text = ("\n" + "\n".join(warnings)) if warnings else ""
            return AgentActionResult(
                False,
                f"Agent run blocked during planning: {error}{warning_text}",
                run,
            )

        warning_text = ("\n" + "\n".join(warnings)) if warnings else ""
        if run.status == "completed":
            return AgentActionResult(
                True,
                f"Agent run {run.run_id} completed.\n\n{run.final_answer}{warning_text}",
                run,
            )
        return AgentActionResult(
            run.status not in ("blocked", "cancelled"),
            f"Agent run {run.run_id} is {run.status}.{warning_text}\n\n"
            f"{self.format_status(run.run_id)}",
            run,
        )

    def status(self, run_id: str = "") -> AgentActionResult:
        run = self._resolve_run(run_id)
        if run is None:
            return AgentActionResult(False, "No agent runs found.")
        return AgentActionResult(True, self.format_status(run.run_id), run)

    def cancel_run(self, run_id: str = "") -> AgentActionResult:
        run = self._resolve_run(run_id)
        if run is None:
            return AgentActionResult(False, "No agent run found.")
        run.status = "cancelled"
        self.store.save(run)
        return AgentActionResult(True, f"Cancelled agent run {run.run_id}.", run)

    def edit_task(self, task_id: str, new_spec: str = "") -> AgentActionResult:
        task_id = task_id.strip()
        if not task_id:
            return AgentActionResult(False, "Usage: /agents edit <task_id> [new input spec]")
        run = self._resolve_run("")
        if run is None:
            return AgentActionResult(False, "No agent run found.")
        task = self._task_by_id(run, task_id)
        if task is None:
            return AgentActionResult(False, f"Task '{task_id}' not found.", run)
        if not new_spec.strip():
            return AgentActionResult(
                True,
                self._format_task_edit_view(task),
                run,
            )
        task.input_spec = new_spec.strip()
        task.status = "pending"
        task.attempts = 0
        task.updated_at = utc_now()
        if run.status in ("blocked", "paused", "completed"):
            run.status = "paused"
        self.store.save(run)
        return AgentActionResult(
            True,
            f"Updated task {task.id}. Run is paused; use /agents run <goal> for a fresh run.",
            run,
        )

    def task_input_spec(self, task_id: str) -> AgentActionResult:
        task_id = task_id.strip()
        if not task_id:
            return AgentActionResult(False, "Task ID required.")
        run = self._resolve_run("")
        if run is None:
            return AgentActionResult(False, "No agent run found.")
        task = self._task_by_id(run, task_id)
        if task is None:
            return AgentActionResult(False, f"Task '{task_id}' not found.", run)
        return AgentActionResult(True, task.input_spec, run)

    def approve_checkpoint(self, checkpoint_id: str) -> AgentActionResult:
        return self._resolve_checkpoint(checkpoint_id, approve=True)

    def reject_checkpoint(self, checkpoint_id: str) -> AgentActionResult:
        return self._resolve_checkpoint(checkpoint_id, approve=False)

    def format_status(self, run_id: str = "") -> str:
        run = self._resolve_run(run_id)
        if run is None:
            return "No agent runs found."
        lines = [
            f"Agent Run: {run.run_id}",
            f"Status: {run.status}",
            f"Goal: {run.goal}",
            "",
            "Tasks:",
        ]
        if not run.tasks:
            lines.append("  (no tasks)")
        for task in run.tasks:
            deps = f" deps={','.join(task.dependencies)}" if task.dependencies else ""
            parent = f" parent={task.parent_task_id}" if task.parent_task_id else ""
            lines.append(
                f"  [{task.id}] {task.role} {task.status} "
                f"attempts={task.attempts}/{task.max_attempts}{deps}{parent}: {task.title}"
            )
        pending = [item for item in run.checkpoints if item.status == "pending"]
        if pending:
            lines.append("")
            lines.append("Pending checkpoints:")
            for item in pending:
                tool = item.tool_call.get("name", "tool")
                lines.append(
                    f"  [{item.checkpoint_id}] {tool} ({item.risk}) "
                    f"for task {item.task_id}"
                )
        if run.final_answer:
            lines.append("")
            lines.append("Final answer:")
            lines.append(run.final_answer)
        return "\n".join(lines)

    def _execute_run(self, run: AgentRun) -> None:
        run.status = "running"
        self.store.save(run)
        for task in dependency_order(run.tasks):
            if task.status == "passed":
                continue
            if any(self._task_status(run, dep) != "passed" for dep in task.dependencies):
                task.status = "blocked"
                run.status = "blocked"
                self.store.save(run)
                return
            self._execute_task(run, task)
            if run.status in ("paused", "blocked", "cancelled"):
                self.store.save(run)
                return

        for _ in range(max(1, self.config.agents.max_iterations)):
            critic = self._run_envelope(
                role="critic",
                task_id="critic",
                run=run,
                messages=critic_messages(run, result_context(run)),
            )
            run.results.append(critic.to_result())
            if critic.status == "pass":
                final = self._run_envelope(
                    role="synthesizer",
                    task_id="final",
                    run=run,
                    messages=synthesizer_messages(run, result_context(run)),
                )
                run.results.append(final.to_result())
                run.final_answer = final.summary
                run.status = "completed"
                self.store.save(run)
                return
            if critic.status == "blocked":
                run.status = "blocked"
                self.store.save(run)
                return
            target = self._revision_target(run, critic.next_actions)
            if target is None:
                run.status = "blocked"
                self.store.save(run)
                return
            target.status = "pending"
            if critic.summary:
                target.input_spec = (
                    target.input_spec + "\n\nCritic feedback:\n" + critic.summary
                ).strip()
            self._execute_task(run, target)
            if run.status in ("paused", "blocked", "cancelled"):
                self.store.save(run)
                return

        run.status = "blocked"
        self.store.save(run)

    def _execute_task(self, run: AgentRun, task: AgentTask) -> None:
        while task.attempts < task.max_attempts:
            task.status = "running"
            task.attempts += 1
            task.updated_at = utc_now()
            self.store.save(run)
            context = self._context_for_task(run, task)
            envelope = self._run_envelope(
                role=task.role,
                task_id=task.id,
                run=run,
                messages=task_messages(run, task, context),
            )
            result = envelope.to_result()
            run.results.append(result)
            paused = self._process_tool_requests(run, task, envelope.tool_requests, result)
            if paused:
                task.status = "paused"
                run.status = "paused"
                self.store.save(run)
                return
            if envelope.status == "pass":
                task.status = "passed"
                self.store.save(run)
                return
            if envelope.status == "blocked":
                task.status = "blocked"
                run.status = "blocked"
                self.store.save(run)
                return
            task.status = "revising"
            self.store.save(run)
        task.status = "blocked"
        run.status = "blocked"
        self.store.save(run)

    def _run_envelope(
        self,
        *,
        role: str,
        task_id: str,
        run: AgentRun,
        messages: list[dict[str, str]],
    ):
        profile = self.config.agents.roles.get(role)
        profile_name = profile.model_profile if profile else self.config.agents.default_profile
        raw = self._completion_text(profile_name, messages)
        try:
            return parse_agent_envelope(
                raw,
                expected_run_id=run.run_id,
                expected_task_id=task_id,
                expected_role=role,
            )
        except AgentProtocolError as error:
            if not self.config.agents.strict_json:
                raise
            repair_raw = self._completion_text(profile_name, repair_prompt(raw, str(error)))
            return parse_agent_envelope(
                repair_raw,
                expected_run_id=run.run_id,
                expected_task_id=task_id,
                expected_role=role,
            )

    def _completion_text(self, profile_name: str, messages: list[dict[str, str]]) -> str:
        response = self.runtime.create_chat_completion_for_profile(
            profile_name,
            messages,
            stream=False,
        )
        if isinstance(response, str):
            return response.strip()
        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def _process_tool_requests(
        self,
        run: AgentRun,
        task: AgentTask,
        requests: list[dict],
        result,
    ) -> bool:
        if not requests:
            return False
        if self.tool_executor_factory is None:
            result.errors.append("Tool requests are not configured for agent runs.")
            return False
        executor = self.tool_executor_factory()
        paused = False
        for item in requests:
            name = str(item.get("name", "")).strip()
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            rationale = str(item.get("rationale", "")).strip()
            call = ToolCall(name=name, args=args, rationale=rationale)
            pending = executor.classify(call)
            if pending.requires_approval and self.config.agents.require_approval:
                checkpoint = AgentCheckpoint(
                    checkpoint_id=pending.call_id,
                    run_id=run.run_id,
                    task_id=task.id,
                    tool_call={
                        "name": call.name,
                        "args": call.args,
                        "rationale": call.rationale,
                    },
                    risk=pending.risk.value,
                )
                run.checkpoints.append(checkpoint)
                paused = True
                continue
            tool_result = executor.execute(pending)
            result.tool_calls.append(self._tool_result_dict(tool_result))
            if not tool_result.success:
                result.errors.append(tool_result.error or "Tool execution failed.")
        return paused

    def _resolve_checkpoint(self, checkpoint_id: str, *, approve: bool) -> AgentActionResult:
        checkpoint_id = checkpoint_id.strip()
        if not checkpoint_id:
            action = "approve" if approve else "reject"
            return AgentActionResult(False, f"Usage: /agents {action} <checkpoint_id>")
        for run in self.store.list_runs():
            for checkpoint in run.checkpoints:
                if checkpoint.checkpoint_id == checkpoint_id or checkpoint.checkpoint_id.startswith(checkpoint_id):
                    if checkpoint.status != "pending":
                        return AgentActionResult(
                            False,
                            f"Checkpoint {checkpoint.checkpoint_id} is already {checkpoint.status}.",
                            run,
                        )
                    if not approve:
                        checkpoint.status = "rejected"
                        checkpoint.resolved_at = utc_now()
                        self.store.save(run)
                        return AgentActionResult(
                            True,
                            f"Rejected checkpoint {checkpoint.checkpoint_id}.",
                            run,
                        )
                    if self.tool_executor_factory is None:
                        return AgentActionResult(False, "Tool executor is not configured.", run)
                    call = ToolCall(
                        name=str(checkpoint.tool_call.get("name", "")),
                        args=dict(checkpoint.tool_call.get("args", {}) or {}),
                        rationale=str(checkpoint.tool_call.get("rationale", "")),
                    )
                    pending = PendingToolCall(
                        call_id=checkpoint.checkpoint_id,
                        call=call,
                        risk=tool_risk(call.name),
                        requires_approval=True,
                    )
                    result = self.tool_executor_factory().execute(pending)
                    checkpoint.status = "executed" if result.success else "failed"
                    checkpoint.result = self._tool_result_dict(result)
                    checkpoint.resolved_at = utc_now()
                    self.store.save(run)
                    status = "executed" if result.success else "failed"
                    return AgentActionResult(
                        result.success,
                        f"Checkpoint {checkpoint.checkpoint_id} {status}: {result.summary()}",
                        run,
                    )
        return AgentActionResult(False, f"Checkpoint '{checkpoint_id}' not found.")

    def _context_for_task(self, run: AgentRun, task: AgentTask) -> str:
        pruning = task.context_pruning or ContextPruning()
        lines: list[str] = []
        if pruning.include_goal:
            lines.append(f"Root goal:\n{run.goal}")
        if pruning.include_ancestors and task.parent_task_id:
            lines.extend(self._ancestor_context(run, task.parent_task_id))
        if pruning.include_dependencies and task.dependencies:
            for dep in task.dependencies:
                for result in run.results:
                    if result.task_id == dep:
                        lines.append(
                            f"Dependency {dep} ({result.role}/{result.status}):\n{result.summary}"
                        )
        text = "\n\n".join(lines).strip()
        if len(text) > pruning.max_context_chars:
            return text[-pruning.max_context_chars :]
        return text

    def _ancestor_context(self, run: AgentRun, parent_id: str) -> list[str]:
        lines: list[str] = []
        task = self._task_by_id(run, parent_id)
        seen: set[str] = set()
        while task is not None and task.id not in seen:
            seen.add(task.id)
            lines.append(f"Ancestor task {task.id}: {task.title}\n{task.instructions}")
            if not task.parent_task_id:
                break
            task = self._task_by_id(run, task.parent_task_id)
        return lines

    def _revision_target(self, run: AgentRun, actions: list[dict]) -> AgentTask | None:
        for action in actions:
            task_id = str(action.get("task_id", "")).strip()
            if task_id:
                task = self._task_by_id(run, task_id)
                if task is not None:
                    feedback = action.get("feedback")
                    if feedback:
                        task.input_spec = (task.input_spec + "\n\n" + str(feedback)).strip()
                    return task
        return None

    def _resolve_run(self, run_id: str) -> AgentRun | None:
        run_id = run_id.strip()
        if run_id:
            return self.store.load(run_id)
        if self.active_run_id:
            run = self.store.load(self.active_run_id)
            if run is not None:
                return run
        return self.store.latest()

    @staticmethod
    def _task_by_id(run: AgentRun, task_id: str) -> AgentTask | None:
        for task in run.tasks:
            if task.id == task_id or task.id.startswith(task_id):
                return task
        return None

    @staticmethod
    def _task_status(run: AgentRun, task_id: str) -> str:
        task = AgentManager._task_by_id(run, task_id)
        return task.status if task is not None else "missing"

    @staticmethod
    def _format_task_edit_view(task: AgentTask) -> str:
        return "\n".join(
            [
                f"Task {task.id}: {task.title}",
                f"Role: {task.role}",
                f"Status: {task.status}",
                f"Parent: {task.parent_task_id or '(none)'}",
                f"Dependencies: {', '.join(task.dependencies) if task.dependencies else '(none)'}",
                "",
                "Instructions:",
                task.instructions,
                "",
                "Current input spec:",
                task.input_spec or "(none)",
                "",
                "To update from CLI: /agents edit "
                f"{task.id} <replacement input spec>",
            ]
        )

    @staticmethod
    def _tool_result_dict(result: ToolResult) -> dict:
        return {
            "call_id": result.call_id,
            "name": result.name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "status": result.status,
        }

    @staticmethod
    def _error_result(message: str):
        from app.agents.models import AgentResult

        return AgentResult(
            task_id="system",
            role="orchestrator",
            status="blocked",
            summary=message,
            errors=[message],
        )
