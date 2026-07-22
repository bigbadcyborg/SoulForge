"""UI-agnostic command router.

Maps a slash-command name + argument string to a ``ChatController`` call and
returns a structured :class:`CommandResult`. The Textual TUI and the CLI each
have their own dispatchers today (``app/tui/app.py`` and ``app/main.py``); this
router is a third, front-end-neutral dispatcher whose first consumer is the
local API server (``app/server``). It deliberately covers the button-friendly
subset of commands — plain request/response actions — and leaves interactive
streaming chat to the caller (see ``ChatController.stream_reply``).

Handlers return text (``kind="message"``), structured data for the GUI to render
in a dialog (``kind="data"``), or an error (``kind="error"``). Long-running
handlers (agent runs, model switches, ingest) run synchronously here; the server
is expected to call :meth:`CommandRouter.dispatch` inside a worker thread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.chat_controller import ChatController
from app.core.commands import format_help_text, help_catalog
from app.core.config import FEATURE_DISPLAY_NAMES


@dataclass
class CommandResult:
    """Structured outcome of a routed command."""

    kind: str  # "message" | "data" | "error"
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    @classmethod
    def message(cls, text: str, *, success: bool = True) -> "CommandResult":
        return cls(kind="message", text=text, success=success)

    @classmethod
    def error(cls, text: str) -> "CommandResult":
        return cls(kind="error", text=text, success=False)

    @classmethod
    def structured(cls, text: str, data: dict[str, Any]) -> "CommandResult":
        return cls(kind="data", text=text, data=data)

    def to_dict(self) -> dict[str, Any]:
        # Coerce text to a string so a handler that forwards a None-returning
        # controller method can never break the API response schema.
        return {
            "kind": self.kind,
            "text": "" if self.text is None else str(self.text),
            "data": self.data,
            "success": self.success,
        }


Handler = Callable[["CommandRouter", str], CommandResult]


class CommandRouter:
    """Routes command names to ChatController calls."""

    def __init__(self, controller: ChatController) -> None:
        self.controller = controller

    # -- public API ------------------------------------------------------

    def command_names(self) -> list[str]:
        return sorted(_HANDLERS.keys())

    def dispatch(self, name: str, args: str = "") -> CommandResult:
        """Route ``name`` (with or without a leading slash) to its handler."""
        key = name.strip().lstrip("/").lower()
        if not key:
            return CommandResult.error("Empty command.")
        handler = _HANDLERS.get(key)
        if handler is None:
            return CommandResult.error(f"Unknown command: {name}")
        try:
            return handler(self, args.strip())
        except Exception as error:  # noqa: BLE001 - surfaced to the caller
            return CommandResult.error(f"{key} failed: {error}")

    # -- handlers --------------------------------------------------------
    # Each handler is small and delegates to a ChatController method. Grouped
    # by area; registered in the _HANDLERS table at the bottom.

    def _help(self, args: str) -> CommandResult:
        if args.strip().lower() == "catalog":
            return CommandResult.structured("Help catalog.", help_catalog())
        return CommandResult.message(format_help_text(args, self.controller.config))

    def _status(self, args: str) -> CommandResult:
        c = self.controller
        return CommandResult.structured(
            f"Model: {c.model_name} | Features: {c.features_summary()}",
            {
                "model": c.model_name,
                "compute_backend": str(c.compute_backend),
                "features": c.features.as_dict(),
                "turn_count": c.turn_count,
            },
        )

    def _health(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.run_health_check())

    def _diagnostics(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.run_diagnostics())

    def _config(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.get_config_view())

    def _reload_soul(self, args: str) -> CommandResult:
        self.controller.reload_soul()
        return CommandResult.message("Reloaded persona from SOUL.md.")

    # features
    def _features(self, args: str) -> CommandResult:
        c = self.controller
        if not args or args.strip().lower() == "data":
            return CommandResult.structured(
                c.features.format_list(), {"features": c.features.as_dict()}
            )
        parts = args.split()
        if len(parts) == 2 and parts[1].lower() in ("on", "off"):
            key, state = parts[0], parts[1].lower() == "on"
            try:
                c.set_feature(key, state)
            except KeyError as error:
                return CommandResult.error(str(error))
            return CommandResult.message(f"{key} set to {'on' if state else 'off'}.")
        return CommandResult.error(
            "Usage: /features | /features <name> on|off. "
            f"Valid: {', '.join(FEATURE_DISPLAY_NAMES.values())}"
        )

    # model routing
    def _model(self, args: str) -> CommandResult:
        c = self.controller
        if not args or args.lower() == "list":
            return CommandResult.structured(
                c.format_model_list(), {"models": c.list_chat_models()}
            )
        name = c.switch_chat_model(args)
        return CommandResult.message(f"Switched to model: {name}")

    def _models(self, args: str) -> CommandResult:
        c = self.controller
        if not args:
            return CommandResult.message(c.format_models_view())
        parts = args.split(maxsplit=2)
        sub = parts[0].lower()
        if sub == "info":
            return CommandResult.structured("Model routing.", c.models_info())
        if sub == "role" and len(parts) == 3:
            return CommandResult.message(c.set_agent_role_model(parts[1], parts[2]))
        if sub == "profile" and len(parts) == 3:
            return CommandResult.message(c.set_agent_profile_model(parts[1], parts[2]))
        if sub == "chat" and len(parts) >= 2:
            return CommandResult.message(
                f"Switched to model: {c.switch_chat_model(parts[1])}"
            )
        if sub == "vision":
            return self._models_vision(parts[1:])
        return CommandResult.error(
            "Usage: /models | /models chat <model> | /models role <role> <model> | "
            "/models profile <profile> <model> | /models vision <model> [mmproj]"
        )

    def _models_vision(self, rest: list[str]) -> CommandResult:
        c = self.controller
        if not rest:
            return CommandResult.message(c.format_vision_view())
        first = rest[0].lower()
        if first in ("off", "disable", "none"):
            return CommandResult.message(c.disable_vision())
        if first == "mmproj" and len(rest) >= 2:
            return CommandResult.message(c.set_vision_mmproj(rest[1]))
        if first == "handler" and len(rest) >= 2:
            v = c.config.vision
            return CommandResult.message(
                c.set_vision_model(v.model_path, v.mmproj_path, handler=rest[1])
            )
        # /models vision <model> [mmproj]
        model = rest[0]
        mmproj = rest[1] if len(rest) >= 2 else None
        return CommandResult.message(c.set_vision_model(model, mmproj))

    # memory
    def _memory(self, args: str) -> CommandResult:
        if args.strip().lower() == "data":
            return CommandResult.structured(
                "Memory sections.", self.controller.memory_sections()
            )
        return CommandResult.message(self.controller.get_memory_view())

    def _memory_set(self, args: str) -> CommandResult:
        # First token = section, remainder (may span lines) = content.
        head, _, content = args.partition("\n")
        section = head.strip().lower()
        if section not in ("user", "memory", "session"):
            return CommandResult.error(
                "Usage: /memory-set <user|memory|session>\\n<content>"
            )
        truncated = self.controller.save_memory(section, content)
        note = " (truncated to the section limit)" if truncated else ""
        return CommandResult.message(f"Saved {section} memory{note}.")

    def _memory_on(self, args: str) -> CommandResult:
        self.controller.enable_memory()
        return CommandResult.message("Memory enabled.")

    def _memory_off(self, args: str) -> CommandResult:
        self.controller.disable_memory()
        return CommandResult.message("Memory disabled.")

    def _memory_clear(self, args: str) -> CommandResult:
        self.controller.clear_all_memory()
        return CommandResult.message(
            "Cleared all memory: user/memory/session files and remembered "
            "conversation history (episodic memory)."
        )

    def _memory_forget(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.clear_episodic_memory())

    def _memory_analysis(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.format_memory_analysis(args))

    def _memory_edit(self, args: str) -> CommandResult:
        section = args.strip().lower()
        if section not in ("user", "memory", "session"):
            return CommandResult.error(
                "Usage: /memory-edit <user|memory|session> (view current content)"
            )
        content = self.controller.memory_manager.read_raw(section)
        return CommandResult.message(content or f"({section} is empty)")

    def _memory_review(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.get_memory_review())

    def _memory_accept(self, args: str) -> CommandResult:
        try:
            self.controller.accept_memory_suggestion()
        except ValueError as error:
            return CommandResult.error(str(error))
        return CommandResult.message("Memory suggestion saved.")

    def _memory_reject(self, args: str) -> CommandResult:
        self.controller.reject_memory_suggestion()
        return CommandResult.message("Memory suggestion discarded.")

    # skills
    def _skills(self, args: str) -> CommandResult:
        mgr = self.controller.skill_manager
        active = mgr.list_skills(status="active")
        archived = mgr.list_skills(status="archived")
        lines = ["Active skills:"]
        lines.extend(
            f"  - {s.get('name')}: {s.get('description', '')}" for s in active
        ) if active else lines.append("  (none)")
        lines.append("Archived skills:")
        lines.extend(
            f"  - {s.get('name')}: {s.get('description', '')}" for s in archived
        ) if archived else lines.append("  (none)")
        return CommandResult.message("\n".join(lines))

    def _success(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.mark_workflow_success(args).message)

    def _crystallize(self, args: str) -> CommandResult:
        result = self.controller.crystallize_workflow()
        return CommandResult.message(result.message or "No workflow ready to crystallize.")

    def _skill_accept(self, args: str) -> CommandResult:
        ok = self.controller.accept_skill_suggestion()
        return CommandResult.message(
            "Skill saved." if ok else "No pending skill suggestion."
        )

    def _skill_reject(self, args: str) -> CommandResult:
        self.controller.reject_skill_suggestion()
        return CommandResult.message("Skill suggestion discarded.")

    def _skill_restore(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /skill-restore <name>")
        return CommandResult.message(self.controller.restore_skill_direct(args).message)

    # curator
    def _curator(self, args: str) -> CommandResult:
        if args.strip().lower() == "data":
            return CommandResult.structured("Curator.", self.controller.curator_data())
        result = self.controller.run_curator_review()
        text = result.message or ""
        if getattr(result, "has_findings", False):
            text = (text + "\n\n" + self.controller.get_curator_review()).strip()
        return CommandResult.message(text or "Curator found nothing to review.")

    def _curator_archive(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /curator-archive <skill name>")
        return CommandResult.message(self.controller.archive_skill_direct(args).message)

    def _curator_compact(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /curator-compact <skill name>")
        result = self.controller.compact_skill_direct(args)
        return CommandResult.message(getattr(result, "message", "Compacted."))

    def _curator_accept(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /curator-accept <finding_id>")
        return CommandResult.message(self.controller.accept_curator_finding(args).message)

    def _curator_ignore(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /curator-ignore <finding_id>")
        self.controller.dismiss_curator_finding(args)
        return CommandResult.message(f"Dismissed curator finding {args}.")

    # tasks
    def _tasks(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.get_board_view())

    def _task_new(self, args: str) -> CommandResult:
        if not args.strip():
            return CommandResult.error("Usage: /task-new <title>")
        return CommandResult.message(self.controller.create_task_direct(args.strip()).message)

    def _task_move(self, args: str) -> CommandResult:
        parts = args.split()
        if len(parts) != 2:
            return CommandResult.error("Usage: /task-move <id> <column>")
        return CommandResult.message(
            self.controller.move_task_direct(parts[0], parts[1]).message
        )

    def _task_done(self, args: str) -> CommandResult:
        if not args.strip():
            return CommandResult.error("Usage: /task-done <id>")
        return CommandResult.message(
            self.controller.move_task_direct(args.strip(), "done").message
        )

    def _task_delete(self, args: str) -> CommandResult:
        if not args.strip():
            return CommandResult.error("Usage: /task-delete <id>")
        return CommandResult.message(self.controller.delete_task_direct(args.strip()).message)

    def _task_suggest(self, args: str) -> CommandResult:
        result = self.controller.run_task_suggest()
        text = result.message or ""
        if getattr(result, "has_suggestions", False):
            text = (text + "\n\n" + self.controller.get_task_suggestions_review()).strip()
        return CommandResult.message(text or "No task suggestions.")

    def _task_accept(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /task-accept <suggestion_id>")
        return CommandResult.message(self.controller.accept_task_suggestion(args).message)

    def _task_reject(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /task-reject <suggestion_id>")
        self.controller.dismiss_task_suggestion(args)
        return CommandResult.message(f"Rejected task suggestion {args}.")

    # tool approvals
    def _tool_approve(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /tool-approve <call_id>")
        return CommandResult.message(self.controller.approve_tool_call(args).message)

    def _tool_reject(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /tool-reject <call_id>")
        return CommandResult.message(self.controller.reject_tool_call(args).message)

    # rag
    def _rag(self, args: str) -> CommandResult:
        c = self.controller
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "on":
            c.enable_rag()
            return CommandResult.message("RAG enabled.")
        if sub == "off":
            c.disable_rag()
            return CommandResult.message("RAG disabled.")
        if sub == "data":
            return CommandResult.structured(
                "RAG data.",
                {"status": c.get_rag_status(), "stats": c.get_rag_stats()},
            )
        if sub == "select":
            if not rest.strip() or rest.strip().lower() == "all":
                c.set_rag_sources(None)
                return CommandResult.message("RAG using all sources.")
            sources = [s.strip() for s in rest.split(",") if s.strip()]
            ok = c.set_rag_sources(sources)
            if not ok:
                return CommandResult.error("Enable RAG first (/rag on).")
            return CommandResult.message(f"RAG using {len(sources)} source(s).")
        if sub == "remove":
            if not rest.strip():
                return CommandResult.error("Usage: /rag remove <filename>")
            return CommandResult.message(c.delete_doc(rest.strip()))
        # Readable status (the old handler returned only the label "RAG status").
        status = c.get_rag_status()
        stats = c.get_rag_stats()
        available = status.get("available_sources") or []
        selected = status.get("selected_sources")
        lines = [
            f"RAG: {'on' if status.get('enabled') else 'off'}",
            f"Indexed chunks: {stats.get('chunk_count', 0)}",
            f"Available sources ({len(available)}):",
        ]
        lines.extend(f"  - {s}" for s in available) if available else lines.append(
            "  (none — run /ingest to index docs/)"
        )
        if selected is not None:
            shown = ", ".join(selected) if selected else "(none)"
            lines.append(f"Selected sources: {shown}")
        lines.append("")
        lines.append("Usage: /rag on | /rag off")
        return CommandResult.structured("\n".join(lines), {"rag": status, "stats": stats})

    def _ingest(self, args: str) -> CommandResult:
        result = self.controller.run_ingest()
        return CommandResult.message(result.summary())

    def _sources(self, args: str) -> CommandResult:
        from app.rag.retriever import Retriever

        return CommandResult.message(
            Retriever.format_sources_detail(self.controller.last_retrieved_chunks)
        )

    # sessions
    def _sessions(self, args: str) -> CommandResult:
        c = self.controller
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "data":
            return CommandResult.structured("Sessions.", c.sessions_data())
        if sub in ("", "list"):
            return CommandResult.message(c.list_sessions_view())
        if sub == "save":
            return CommandResult.message(c.save_session_direct(rest).message)
        if sub == "load":
            return CommandResult.message(c.load_session_direct(rest).message)
        if sub == "delete":
            return CommandResult.message(c.delete_session_direct(rest).message)
        if sub == "summary":
            return CommandResult.message(c.run_session_summary().message)
        return CommandResult.error(
            "Usage: /sessions list|save|load|delete|summary [arg]"
        )

    def _session_save(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.save_session_direct(args).message)

    def _session_load(self, args: str) -> CommandResult:
        if not args:
            return CommandResult.error("Usage: /session-load <id>")
        return CommandResult.message(self.controller.load_session_direct(args).message)

    def _session_summary(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.run_session_summary().message)

    # agents
    def _agents(self, args: str) -> CommandResult:
        c = self.controller
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "data":
            return CommandResult.structured("Agents.", c.agents_data(rest))
        if sub in ("", "status"):
            return CommandResult.message(c.get_agents_status(rest).message)
        if sub in ("on", "off"):
            c.set_feature("agents", sub == "on")
            return CommandResult.message(f"Agents set to {sub}.")
        if sub == "load":
            return CommandResult.message(c.preload_agent_models())
        if sub == "run":
            return CommandResult.message(c.run_agent_workflow(rest).message)
        if sub == "resume":
            return CommandResult.message(c.resume_agent_run(rest).message)
        if sub == "cancel":
            return CommandResult.message(c.cancel_agent_run(rest).message)
        if sub == "approve":
            return CommandResult.message(c.approve_agent_checkpoint(rest).message)
        if sub == "reject":
            return CommandResult.message(c.reject_agent_checkpoint(rest).message)
        if sub == "edit":
            edit = rest.split(maxsplit=1)
            task_id = edit[0] if edit else ""
            spec = edit[1] if len(edit) > 1 else ""
            return CommandResult.message(c.edit_agent_task(task_id, spec).message)
        return CommandResult.error(
            "Usage: /agents status|on|off|load|run|resume|cancel|approve|reject|edit [arg]"
        )

    # tools
    def _tools(self, args: str) -> CommandResult:
        c = self.controller
        if not args:
            return CommandResult.message(c.get_tools_status())
        parts = args.split(maxsplit=1)
        sub = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "data":
            return CommandResult.structured("Tools.", c.get_tools_menu_data())
        if sub == "allowlist":
            return CommandResult.structured(
                c.get_tools_status(),
                {"shellAllowlist": list(c.config.tools.shell_allowlist)},
            )
        if sub == "add-shell":
            return CommandResult.message(c.add_shell_allowlist_entry(rest))
        if sub == "remove-shell":
            return CommandResult.message(c.remove_shell_allowlist_entry(rest))
        if sub == "allow":
            # /tools allow shell|write|network on|off
            parts_allow = rest.split()
            if len(parts_allow) != 2 or parts_allow[1].lower() not in ("on", "off"):
                return CommandResult.error(
                    "Usage: /tools allow shell|write|network on|off"
                )
            try:
                return CommandResult.message(
                    c.set_tool_permission(parts_allow[0], parts_allow[1].lower() == "on")
                )
            except ValueError as error:
                return CommandResult.error(str(error))
        if sub == "test":
            name_parts = rest.split(maxsplit=1)
            if not name_parts:
                return CommandResult.error("Usage: /tools test <name> '<json args>'")
            name = name_parts[0]
            raw = name_parts[1] if len(name_parts) > 1 else "{}"
            try:
                tool_args = json.loads(raw)
            except json.JSONDecodeError as error:
                return CommandResult.error(f"Invalid JSON: {error}")
            if not isinstance(tool_args, dict):
                return CommandResult.error("Tool args must be a JSON object.")
            result = c.run_tool_test(name, tool_args)
            status = "OK" if result.success else "FAILED"
            return CommandResult.message(f"[{status}] {result.summary(4000)}")
        return CommandResult.error(
            "Usage: /tools | /tools allowlist | /tools add-shell <cmd> | "
            "/tools remove-shell <cmd> | /tools allow shell|write|network on|off | "
            "/tools test <name> '<json>'"
        )

    def _tools_log(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.get_tool_log_view())

    # kanban structured board
    def _kanban(self, args: str) -> CommandResult:
        return CommandResult.structured("Kanban board.", self.controller.board_data())


_HANDLERS: dict[str, Handler] = {
    # general
    "help": CommandRouter._help,
    "status": CommandRouter._status,
    "health": CommandRouter._health,
    "diagnostics": CommandRouter._diagnostics,
    "config": CommandRouter._config,
    "reload-soul": CommandRouter._reload_soul,
    "features": CommandRouter._features,
    # model routing
    "model": CommandRouter._model,
    "models": CommandRouter._models,
    # rag
    "rag": CommandRouter._rag,
    "ingest": CommandRouter._ingest,
    "sources": CommandRouter._sources,
    # memory
    "memory": CommandRouter._memory,
    "memory-on": CommandRouter._memory_on,
    "memory-off": CommandRouter._memory_off,
    "memory-clear": CommandRouter._memory_clear,
    "memory-forget": CommandRouter._memory_forget,
    "memory-set": CommandRouter._memory_set,
    "memory-analysis": CommandRouter._memory_analysis,
    "memory-search": CommandRouter._memory_analysis,
    "memory-edit": CommandRouter._memory_edit,
    "memory-review": CommandRouter._memory_review,
    "memory-accept": CommandRouter._memory_accept,
    "memory-reject": CommandRouter._memory_reject,
    # skills
    "skills": CommandRouter._skills,
    "success": CommandRouter._success,
    "crystallize": CommandRouter._crystallize,
    "skill-accept": CommandRouter._skill_accept,
    "skill-reject": CommandRouter._skill_reject,
    "skill-restore": CommandRouter._skill_restore,
    # curator
    "curator": CommandRouter._curator,
    "curator-review": CommandRouter._curator,
    "curator-archive": CommandRouter._curator_archive,
    "curator-compact": CommandRouter._curator_compact,
    "curator-accept": CommandRouter._curator_accept,
    "curator-ignore": CommandRouter._curator_ignore,
    # tasks
    "tasks": CommandRouter._tasks,
    "kanban": CommandRouter._kanban,
    "task-new": CommandRouter._task_new,
    "task-move": CommandRouter._task_move,
    "task-done": CommandRouter._task_done,
    "task-delete": CommandRouter._task_delete,
    "task-suggest": CommandRouter._task_suggest,
    "task-accept": CommandRouter._task_accept,
    "task-reject": CommandRouter._task_reject,
    # agents
    "agents": CommandRouter._agents,
    # sessions
    "sessions": CommandRouter._sessions,
    "session-list": CommandRouter._sessions,
    "session-save": CommandRouter._session_save,
    "session-load": CommandRouter._session_load,
    "session-summary": CommandRouter._session_summary,
    # tools
    "tools": CommandRouter._tools,
    "tools-log": CommandRouter._tools_log,
    "tool-approve": CommandRouter._tool_approve,
    "tool-reject": CommandRouter._tool_reject,
}
