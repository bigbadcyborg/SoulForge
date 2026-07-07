"""Agent orchestration manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from app.agents.json_protocol import (
    AgentProtocolError,
    dependency_order,
    parse_agent_envelope,
    repair_prompt,
    tasks_from_planner_envelope,
)
from app.agents.models import (
    AgentCheckpoint,
    AgentJsonEnvelope,
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.memory.memory_manager import MemoryManager
    from app.rag.retriever import Retriever
    from app.skills.skill_manager import SkillManager

ToolExecutorFactory = Callable[[], ToolExecutor]
RetrieverProvider = Callable[[], "Retriever | None"]
ProgressCallback = Callable[[str], None]

# Roles that receive retrieved RAG context by default. The researcher exists to
# gather local context, so it is the natural recipient; other roles can still
# opt in via the task's context_pruning.include_rag flag.
_RAG_DEFAULT_ROLES: frozenset[str] = frozenset({"researcher"})


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
        retriever_provider: RetrieverProvider | None = None,
        memory_manager: "MemoryManager | None" = None,
        skill_manager: "SkillManager | None" = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.store = AgentRunStore(config)
        self.tool_executor_factory = tool_executor_factory
        self.retriever_provider = retriever_provider
        self.memory_manager = memory_manager
        self.skill_manager = skill_manager
        self.on_progress: ProgressCallback | None = None
        self.active_run_id: str = ""

    def _emit(self, line: str) -> None:
        """Report a short progress line, never letting the callback break a run."""
        if self.on_progress is None:
            return
        try:
            self.on_progress(line)
        except Exception:  # noqa: BLE001 - progress is best-effort
            pass

    def start_run(
        self, goal: str, *, on_progress: ProgressCallback | None = None
    ) -> AgentActionResult:
        goal = goal.strip()
        if not goal:
            return AgentActionResult(False, "Usage: /agents run <goal>")

        self.on_progress = on_progress
        try:
            run = AgentRun(run_id=new_id("run"), goal=goal, status="running")
            self.active_run_id = run.run_id
            self.store.save(run)

            warnings: list[str] = []
            try:
                self._emit("● orchestrator: planning task graph")
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
                self._emit(f"● plan ready: {len(run.tasks)} task(s)")
                self.store.save(run)
                # Warm residents only after planning: the orchestrator's swap
                # profile evicts every other loaded profile, so warming earlier
                # would load the worker models just to throw them away.
                warnings = self.runtime.warm_resident_profiles()
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

            return self._run_outcome(run, warnings)
        finally:
            self.on_progress = None

    def resume_run(
        self, run_id: str = "", *, on_progress: ProgressCallback | None = None
    ) -> AgentActionResult:
        """Continue a paused run after checkpoints are resolved or a task is edited."""
        run = self._resolve_run(run_id)
        if run is None:
            return AgentActionResult(False, "No agent run found.")
        if run.status not in ("paused", "blocked"):
            return AgentActionResult(
                False,
                f"Run {run.run_id} is {run.status}; only paused or blocked runs can be resumed.",
                run,
            )
        pending = [item for item in run.checkpoints if item.status == "pending"]
        if pending:
            ids = ", ".join(item.checkpoint_id for item in pending)
            return AgentActionResult(
                False,
                f"Resolve pending checkpoint(s) first: {ids}",
                run,
            )

        self.on_progress = on_progress
        try:
            self.active_run_id = run.run_id
            for task in run.tasks:
                if task.status in ("paused", "blocked", "revising", "running"):
                    task.status = "pending"
                    task.attempts = 0
                    task.updated_at = utc_now()

            self._emit(f"● resuming run {run.run_id}")
            warnings = self.runtime.warm_resident_profiles()
            try:
                self._execute_run(run)
            except Exception as error:  # noqa: BLE001
                run.status = "blocked"
                run.results.append(self._error_result(str(error)))
                self.store.save(run)
                warning_text = ("\n" + "\n".join(warnings)) if warnings else ""
                return AgentActionResult(
                    False,
                    f"Agent run blocked while resuming: {error}{warning_text}",
                    run,
                )

            return self._run_outcome(run, warnings)
        finally:
            self.on_progress = None

    def _run_outcome(self, run: AgentRun, warnings: list[str]) -> AgentActionResult:
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
        task = self._task_by_id(run, task_id, allow_prefix=True)
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
            f"Updated task {task.id}. Run is paused; use /agents resume to continue with the new spec.",
            run,
        )

    def task_input_spec(self, task_id: str) -> AgentActionResult:
        task_id = task_id.strip()
        if not task_id:
            return AgentActionResult(False, "Task ID required.")
        run = self._resolve_run("")
        if run is None:
            return AgentActionResult(False, "No agent run found.")
        task = self._task_by_id(run, task_id, allow_prefix=True)
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
            self._emit("● critic: reviewing run against the goal")
            critic = self._run_envelope(
                role="critic",
                task_id="critic",
                run=run,
                messages=critic_messages(run, result_context(run)),
            )
            run.results.append(critic.to_result())
            self._emit(f"● critic verdict: {critic.status}")
            if critic.status == "pass":
                self._emit("● synthesizer: composing final answer")
                final = self._run_envelope(
                    role="synthesizer",
                    task_id="final",
                    run=run,
                    messages=synthesizer_messages(run, result_context(run)),
                )
                run.results.append(final.to_result())
                run.final_answer = final.summary
                run.status = "completed"
                self._emit("✓ run completed")
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
            self._emit(f"↻ revising {target.role}/{target.id}: {target.title}")
            target.status = "pending"
            # A critic-requested revision grants fresh attempts; otherwise a
            # task that passed on its final attempt would re-block instantly.
            target.attempts = 0
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
            self._emit(
                f"▶ {task.role}/{task.id} "
                f"(attempt {task.attempts}/{task.max_attempts}): {task.title}"
            )
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
                self._emit(f"⏸ {task.role}/{task.id} paused for tool approval")
                self.store.save(run)
                return
            if envelope.status == "pass":
                task.status = "passed"
                self._emit(f"✓ {task.role}/{task.id} passed")
                self.store.save(run)
                return
            if envelope.status == "blocked":
                task.status = "blocked"
                run.status = "blocked"
                self._emit(f"✗ {task.role}/{task.id} blocked")
                self.store.save(run)
                return
            task.status = "revising"
            self.store.save(run)
        task.status = "blocked"
        run.status = "blocked"
        self._emit(f"✗ {task.role}/{task.id} blocked (max attempts)")
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
            repair_raw = self._completion_text(profile_name, repair_prompt(raw, str(error)))
            try:
                return parse_agent_envelope(
                    repair_raw,
                    expected_run_id=run.run_id,
                    expected_task_id=task_id,
                    expected_role=role,
                )
            except AgentProtocolError:
                if self.config.agents.strict_json:
                    raise
                # Lenient mode: accept the raw text as the task result rather
                # than blocking the run on a schema failure.
                text = (repair_raw or raw).strip()
                return AgentJsonEnvelope(
                    schema_version=1,
                    role=role,
                    run_id=run.run_id,
                    task_id=task_id,
                    parent_task_id=None,
                    context_pruning=ContextPruning(),
                    status="pass",
                    summary=text,
                    errors=[
                        "Response was not valid envelope JSON; "
                        "accepted as plain text because agents.strictJson is false."
                    ],
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
        role_config = self.config.agents.roles.get(task.role)
        allowed = list(role_config.allowed_tools) if role_config else []
        paused = False
        for item in requests:
            name = str(item.get("name", "")).strip()
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            rationale = str(item.get("rationale", "")).strip()
            # An empty allowlist means "no per-role restriction". A non-empty
            # allowlist scopes the role to exactly those tools; anything else is
            # refused before it can run or become a checkpoint.
            if allowed and name not in allowed:
                message = (
                    f"Tool '{name}' not allowed for role '{task.role}'. "
                    f"Allowed: {', '.join(allowed)}."
                )
                result.errors.append(message)
                self._emit(f"  ✗ {task.role} tool blocked: {name}")
                continue
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

    def _find_checkpoint(
        self, checkpoint_id: str
    ) -> tuple[AgentRun, AgentCheckpoint] | str | None:
        """Locate a checkpoint by exact id, or unique prefix as a fallback.

        Returns the (run, checkpoint) pair, an error message for ambiguous
        prefixes, or None when nothing matches.
        """
        runs = self.store.list_runs()
        for run in runs:
            for checkpoint in run.checkpoints:
                if checkpoint.checkpoint_id == checkpoint_id:
                    return run, checkpoint
        matches = [
            (run, checkpoint)
            for run in runs
            for checkpoint in run.checkpoints
            if checkpoint.checkpoint_id.startswith(checkpoint_id)
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            ids = ", ".join(checkpoint.checkpoint_id for _, checkpoint in matches)
            return f"Checkpoint id '{checkpoint_id}' is ambiguous. Matches: {ids}"
        return None

    def _resolve_checkpoint(self, checkpoint_id: str, *, approve: bool) -> AgentActionResult:
        checkpoint_id = checkpoint_id.strip()
        if not checkpoint_id:
            action = "approve" if approve else "reject"
            return AgentActionResult(False, f"Usage: /agents {action} <checkpoint_id>")
        found = self._find_checkpoint(checkpoint_id)
        if found is None:
            return AgentActionResult(False, f"Checkpoint '{checkpoint_id}' not found.")
        if isinstance(found, str):
            return AgentActionResult(False, found)
        run, checkpoint = found
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
                f"Rejected checkpoint {checkpoint.checkpoint_id}."
                f"{self._resume_hint(run)}",
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
            f"Checkpoint {checkpoint.checkpoint_id} {status}: {result.summary()}"
            f"{self._resume_hint(run)}",
            run,
        )

    @staticmethod
    def _resume_hint(run: AgentRun) -> str:
        if run.status != "paused":
            return ""
        if any(item.status == "pending" for item in run.checkpoints):
            return " More checkpoints are pending; resolve them, then /agents resume."
        return " Run /agents resume to continue the run."

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
        lines.extend(self._local_context(run, task, pruning))
        for checkpoint in run.checkpoints:
            if checkpoint.task_id != task.id or checkpoint.status == "pending":
                continue
            tool = checkpoint.tool_call.get("name", "tool")
            if checkpoint.status == "rejected":
                lines.append(
                    f"Tool request '{tool}' (checkpoint {checkpoint.checkpoint_id}) "
                    "was rejected by the user. Do not request it again; adapt or block."
                )
            else:
                output = ""
                if checkpoint.result:
                    output = str(
                        checkpoint.result.get("output")
                        or checkpoint.result.get("error")
                        or ""
                    )
                lines.append(
                    f"Tool '{tool}' (checkpoint {checkpoint.checkpoint_id}) "
                    f"{checkpoint.status}:\n{output}"
                )
        text = "\n\n".join(lines).strip()
        if len(text) > pruning.max_context_chars:
            return text[-pruning.max_context_chars :]
        return text

    def _local_context(
        self, run: AgentRun, task: AgentTask, pruning: ContextPruning
    ) -> list[str]:
        """Inject retrieved RAG chunks, memory files, and the skill index.

        Every source is best-effort: a missing vector store, unreadable memory
        file, or empty registry must never block a run.
        """
        lines: list[str] = []
        want_rag = pruning.include_rag or task.role in _RAG_DEFAULT_ROLES
        if want_rag:
            rag = self._rag_context(task)
            if rag:
                lines.append(rag)
        if pruning.include_memory:
            memory = self._memory_context()
            if memory:
                lines.append(memory)
        if pruning.include_skills:
            skills = self._skill_index()
            if skills:
                lines.append(skills)
        return lines

    def _rag_context(self, task: AgentTask) -> str:
        if self.retriever_provider is None:
            return ""
        try:
            retriever = self.retriever_provider()
            if retriever is None:
                return ""
            query = f"{task.title}\n{task.instructions}".strip()
            chunks = retriever.retrieve(query)
            if not chunks:
                return ""
            from app.rag.retriever import Retriever

            return "Retrieved documents (RAG):\n" + Retriever.format_context(chunks)
        except Exception as error:  # noqa: BLE001 - RAG is supplementary
            return f"(RAG retrieval unavailable: {error})"

    def _memory_context(self) -> str:
        if self.memory_manager is None:
            return ""
        try:
            snapshot = self.memory_manager.load()
        except Exception:  # noqa: BLE001 - memory is supplementary
            return ""
        if snapshot.is_empty:
            return ""
        parts = []
        if snapshot.user:
            parts.append(f"User facts:\n{snapshot.user}")
        if snapshot.memory:
            parts.append(f"Project memory:\n{snapshot.memory}")
        if snapshot.session:
            parts.append(f"Session notes:\n{snapshot.session}")
        return "Local memory:\n" + "\n\n".join(parts)

    def _skill_index(self) -> str:
        if self.skill_manager is None:
            return ""
        try:
            skills = self.skill_manager.list_skills(status="active")
        except Exception:  # noqa: BLE001 - skills are supplementary
            return ""
        if not skills:
            return ""
        entries = []
        for meta in skills:
            name = str(meta.get("name", "")).strip()
            if not name:
                continue
            desc = str(meta.get("description") or meta.get("trigger") or "").strip()
            entries.append(f"- {name}: {desc}" if desc else f"- {name}")
        if not entries:
            return ""
        return "Available skills (request via tools if needed):\n" + "\n".join(entries)

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
    def _task_by_id(
        run: AgentRun,
        task_id: str,
        *,
        allow_prefix: bool = False,
    ) -> AgentTask | None:
        """Look up a task by exact id; unique-prefix match is CLI convenience only.

        Internal callers (dependency gating, ancestor chains, critic targets)
        must use exact ids — a prefix like 'task_1' would otherwise match
        'task_10'.
        """
        for task in run.tasks:
            if task.id == task_id:
                return task
        if not allow_prefix:
            return None
        matches = [task for task in run.tasks if task.id.startswith(task_id)]
        return matches[0] if len(matches) == 1 else None

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
