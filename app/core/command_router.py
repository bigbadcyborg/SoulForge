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

from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.chat_controller import ChatController
from app.core.commands import format_help_text
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
        return {
            "kind": self.kind,
            "text": self.text,
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
        return CommandResult.message(self.controller.reload_soul())

    # features
    def _features(self, args: str) -> CommandResult:
        c = self.controller
        if not args:
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
        if sub == "role" and len(parts) == 3:
            return CommandResult.message(c.set_agent_role_model(parts[1], parts[2]))
        if sub == "profile" and len(parts) == 3:
            return CommandResult.message(c.set_agent_profile_model(parts[1], parts[2]))
        if sub == "chat" and len(parts) >= 2:
            return CommandResult.message(
                f"Switched to model: {c.switch_chat_model(parts[1])}"
            )
        return CommandResult.error(
            "Usage: /models | /models chat <model> | /models role <role> <model> | "
            "/models profile <profile> <model>"
        )

    # memory
    def _memory(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.get_memory_view())

    def _memory_on(self, args: str) -> CommandResult:
        self.controller.enable_memory()
        return CommandResult.message("Memory enabled.")

    def _memory_off(self, args: str) -> CommandResult:
        self.controller.disable_memory()
        return CommandResult.message("Memory disabled.")

    def _memory_clear(self, args: str) -> CommandResult:
        self.controller.clear_all_memory()
        return CommandResult.message("Cleared all memory sections.")

    # rag
    def _rag(self, args: str) -> CommandResult:
        c = self.controller
        sub = args.split()[0].lower() if args else ""
        if sub == "on":
            c.enable_rag()
            return CommandResult.message("RAG enabled.")
        if sub == "off":
            c.disable_rag()
            return CommandResult.message("RAG disabled.")
        stats = c.get_rag_stats()
        return CommandResult.structured("RAG status.", {"rag": stats})

    # sessions
    def _sessions(self, args: str) -> CommandResult:
        c = self.controller
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
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

    # agents
    def _agents(self, args: str) -> CommandResult:
        c = self.controller
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""
        if sub in ("", "status"):
            return CommandResult.message(c.get_agents_status(rest).message)
        if sub in ("on", "off"):
            c.set_feature("agents", sub == "on")
            return CommandResult.message(f"Agents set to {sub}.")
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
            "Usage: /agents status|on|off|run|resume|cancel|approve|reject|edit [arg]"
        )

    # tools
    def _tools(self, args: str) -> CommandResult:
        c = self.controller
        if not args:
            return CommandResult.message(c.get_tools_status())
        parts = args.split(maxsplit=1)
        sub = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "allowlist":
            return CommandResult.structured(
                c.get_tools_status(),
                {"shellAllowlist": list(c.config.tools.shell_allowlist)},
            )
        if sub == "add-shell":
            return CommandResult.message(c.add_shell_allowlist_entry(rest))
        return CommandResult.error("Usage: /tools | /tools allowlist | /tools add-shell <cmd>")

    def _tools_log(self, args: str) -> CommandResult:
        return CommandResult.message(self.controller.get_tool_log_view())


_HANDLERS: dict[str, Handler] = {
    "help": CommandRouter._help,
    "status": CommandRouter._status,
    "health": CommandRouter._health,
    "diagnostics": CommandRouter._diagnostics,
    "config": CommandRouter._config,
    "reload-soul": CommandRouter._reload_soul,
    "features": CommandRouter._features,
    "model": CommandRouter._model,
    "models": CommandRouter._models,
    "memory": CommandRouter._memory,
    "memory-on": CommandRouter._memory_on,
    "memory-off": CommandRouter._memory_off,
    "memory-clear": CommandRouter._memory_clear,
    "rag": CommandRouter._rag,
    "sessions": CommandRouter._sessions,
    "session-list": CommandRouter._sessions,
    "agents": CommandRouter._agents,
    "tools": CommandRouter._tools,
    "tools-log": CommandRouter._tools_log,
}
