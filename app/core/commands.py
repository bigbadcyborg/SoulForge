"""Shared command help registry for TUI and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import AppConfig


@dataclass(frozen=True)
class CommandHelp:
    category: str
    usage: str
    description: str


COMMANDS: tuple[CommandHelp, ...] = (
    CommandHelp(
        "General",
        "/help [topic]",
        "Show commands; use /help models, /help agents, or /help crystallize",
    ),
    CommandHelp("General", "/status", "Show model name, active features, and app state"),
    CommandHelp(
        "Model",
        "/model",
        "Open model picker (TUI) or show current model (CLI)",
    ),
    CommandHelp(
        "Model",
        "/model list",
        "List chat models in ./models/ (* marks current)",
    ),
    CommandHelp(
        "Model",
        "/model add <path>",
        "Import a .gguf file into ./models/ by copying it",
    ),
    CommandHelp(
        "Model",
        "/model add <path> switch",
        "Import a .gguf and switch to it immediately",
    ),
    CommandHelp(
        "Model",
        "/model <name>",
        "Switch to a chat model by filename or unique partial match",
    ),
    CommandHelp(
        "Model",
        "/models",
        "Show chat model and per-agent role/profile model routing",
    ),
    CommandHelp(
        "Model",
        "/models help",
        "Show the detailed model routing guide",
    ),
    CommandHelp(
        "Model",
        "/models chat <model>",
        "Switch the normal chat model by filename or unique partial match",
    ),
    CommandHelp(
        "Model",
        "/models add <path> [switch]",
        "Alias for /model add <path> [switch]",
    ),
    CommandHelp(
        "Model",
        "/models role <role> <model|inherit>",
        "Assign one agent role to a model without changing sibling roles",
    ),
    CommandHelp(
        "Model",
        "/models profile <profile> <model|inherit>",
        "Change a shared agent model profile used by one or more roles",
    ),
    CommandHelp(
        "General",
        "/health",
        "Short pass/warn/fail summary with top remediation hints",
    ),
    CommandHelp(
        "General",
        "/diagnostics",
        "Full system check: models, CUDA, RAG, skills, kanban (TUI modal)",
    ),
    CommandHelp(
        "General",
        "/config",
        "Resolved configuration paths, features, and limits (read-only)",
    ),
    CommandHelp(
        "General",
        "/tutorial",
        "Open the onboarding tutorial wizard (TUI)",
    ),
    CommandHelp(
        "Tools",
        "/tools",
        "Open tools workshop (TUI) or list status (CLI)",
    ),
    CommandHelp(
        "Tools",
        "/tools test <name> '<json>'",
        "Run a manual tool test (CLI)",
    ),
    CommandHelp(
        "Tools",
        "/tools add-shell <command>",
        "Add a shell command prefix to shellAllowlist",
    ),
    CommandHelp(
        "Tools",
        "/tools allowlist",
        "List shellAllowlist entries",
    ),
    CommandHelp(
        "Tools",
        "/tools-log",
        "Show recent tool call audit log entries",
    ),
    CommandHelp(
        "Testing",
        "/simulate [type|list|all]",
        "Run built-in red-team simulation payloads and show pass/fail results",
    ),
    CommandHelp(
        "Tools",
        "/tool-approve <id>",
        "Approve and execute a pending tool call (CLI)",
    ),
    CommandHelp(
        "Tools",
        "/tool-reject <id>",
        "Reject a pending tool call (CLI)",
    ),
    CommandHelp("General", "/exit, /quit", "Exit the application"),
    CommandHelp(
        "Features",
        "/features",
        "Open the feature toggle menu (TUI); changes auto-save to config.yaml",
    ),
    CommandHelp(
        "Features",
        "/features list",
        "List all feature flags and their on/off state",
    ),
    CommandHelp(
        "Features",
        "/features <name> on|off",
        "Toggle one feature (e.g. /features rag on, /features soul off)",
    ),
    CommandHelp(
        "RAG",
        "/ingest",
        "Index files from docs/ into ChromaDB (text + PDF/OCR)",
    ),
    CommandHelp(
        "RAG",
        "/sources",
        "View retrieved chunks from the last question",
    ),
    CommandHelp(
        "RAG",
        "/rag",
        "Toggle RAG or open document selection modal (TUI)",
    ),
    CommandHelp("RAG", "/rag on", "Enable RAG retrieval"),
    CommandHelp("RAG", "/rag off", "Disable RAG retrieval"),
    CommandHelp("RAG", "/rag all", "Enable RAG using all indexed documents"),
    CommandHelp(
        "RAG",
        "/rag doc1,doc2",
        "Enable RAG filtered to specific document names",
    ),
    CommandHelp(
        "Persona",
        "/reload-soul",
        "Reload SOUL.md persona into the system prompt",
    ),
    CommandHelp(
        "Memory",
        "/memory",
        "View user.md, memory.md, and session.md with character counts",
    ),
    CommandHelp(
        "Memory",
        "/memory-analysis [query]",
        "Analyze static memory and search vector-backed episodic memory",
    ),
    CommandHelp(
        "Memory",
        "/memory-edit [name]",
        "Edit user, memory, or session (TUI editor; CLI: optional inline content)",
    ),
    CommandHelp(
        "Memory",
        "/memory-on",
        "Enable memory injection (equivalent to /features memory on)",
    ),
    CommandHelp(
        "Memory",
        "/memory-off",
        "Disable memory injection (equivalent to /features memory off)",
    ),
    CommandHelp(
        "Memory",
        "/memory-clear",
        "Wipe user.md, memory.md, and session.md",
    ),
    CommandHelp(
        "Memory",
        "/memory-review",
        "Show pending memory update suggestion",
    ),
    CommandHelp(
        "Memory",
        "/memory-accept",
        "Save the pending suggestion to user.md or memory.md",
    ),
    CommandHelp(
        "Memory",
        "/memory-reject",
        "Discard the pending memory suggestion",
    ),
    CommandHelp(
        "Skills",
        "/skills",
        "Browse active skills, create manually, or archive (TUI modal)",
    ),
    CommandHelp(
        "Skills",
        "/success [note]",
        "Mark the recent chat workflow as successful (counts toward crystallization)",
    ),
    CommandHelp(
        "Skills",
        "/crystallize [fingerprint]",
        "Draft a reusable skill from a workflow that reached the success threshold",
    ),
    CommandHelp(
        "Skills",
        "/skill-accept",
        "Save the pending skill draft to app/skills/active/ (requires skills on)",
    ),
    CommandHelp(
        "Skills",
        "/skill-reject",
        "Discard the pending skill draft without saving",
    ),
    CommandHelp(
        "Skills",
        "/skill-restore <name>",
        "Restore an archived skill to app/skills/active/",
    ),
    CommandHelp(
        "Curator",
        "/curator",
        "Open curator review modal for pending findings (TUI)",
    ),
    CommandHelp(
        "Curator",
        "/curator-review",
        "Scan active skills for stale, bloated, or duplicate entries",
    ),
    CommandHelp(
        "Curator",
        "/curator-archive <skill>",
        "Archive a skill immediately (moves to skills/archived/)",
    ),
    CommandHelp(
        "Curator",
        "/curator-compact <skill>",
        "Draft a shorter version of a bloated skill for approval",
    ),
    CommandHelp(
        "Curator",
        "/curator-accept <id>",
        "Apply a curator finding by ID (CLI; TUI uses Approve in modal)",
    ),
    CommandHelp(
        "Curator",
        "/curator-ignore <id>",
        "Dismiss a curator finding by ID (CLI)",
    ),
    CommandHelp(
        "Kanban",
        "/tasks",
        "Open Kanban board modal (TUI) or print board (CLI)",
    ),
    CommandHelp(
        "Kanban",
        "/task-new [title]",
        "Create a task in Backlog (modal if no title in TUI)",
    ),
    CommandHelp(
        "Kanban",
        "/task-move <id> <column>",
        "Move a task (backlog, in_progress, blocked, done)",
    ),
    CommandHelp(
        "Kanban",
        "/task-done <id>",
        "Move a task to Done",
    ),
    CommandHelp(
        "Kanban",
        "/task-delete <id>",
        "Delete a task from the board",
    ),
    CommandHelp(
        "Kanban",
        "/task-suggest",
        "Suggest task updates from recent conversation (requires approval)",
    ),
    CommandHelp(
        "Kanban",
        "/task-accept <id>",
        "Apply a pending task suggestion by ID (CLI; TUI uses Approve in modal)",
    ),
    CommandHelp(
        "Kanban",
        "/task-reject <id>",
        "Dismiss a pending task suggestion by ID (CLI)",
    ),
    CommandHelp(
        "Agents",
        "/agents",
        "Show multi-agent run status and model profile residency",
    ),
    CommandHelp(
        "Agents",
        "/agents help",
        "Show the detailed multi-agent workflow guide",
    ),
    CommandHelp(
        "Agents",
        "/agents on|off",
        "Enable or disable opt-in multi-agent workflows",
    ),
    CommandHelp(
        "Agents",
        "/agents run <goal>",
        "Run the Orchestrator/worker pipeline for a goal",
    ),
    CommandHelp(
        "Agents",
        "/agents status [run_id]",
        "Show a saved agent run and pending checkpoints",
    ),
    CommandHelp(
        "Agents",
        "/agents edit <task_id> [new spec]",
        "View or replace a task input spec when a run gets stuck",
    ),
    CommandHelp(
        "Agents",
        "/agents approve|reject <checkpoint_id>",
        "Resolve an agent tool checkpoint",
    ),
    CommandHelp(
        "Agents",
        "/agents cancel [run_id]",
        "Cancel the active or selected agent run",
    ),
    CommandHelp(
        "Sessions",
        "/session-list",
        "List saved conversations (TUI opens browser modal)",
    ),
    CommandHelp(
        "Sessions",
        "/session-save [title]",
        "Save current chat to app/sessions/ (modal in TUI if no title)",
    ),
    CommandHelp(
        "Sessions",
        "/session-load <id>",
        "Restore a saved conversation and rebuild the chat view",
    ),
    CommandHelp(
        "Sessions",
        "/session-summary",
        "LLM summary of current chat written to session.md for prompt injection",
    ),
)


def _skill_settings(config: AppConfig | None) -> tuple[int, int, bool]:
    if config is None:
        return 3, 3, False
    skills = config.skills
    return (
        skills.min_successful_repeats,
        skills.success_window_turns,
        skills.auto_create,
    )


def _skill_crystallization_summary(config: AppConfig | None = None) -> str:
    min_repeats, window_turns, auto_create = _skill_settings(config)
    auto_line = (
        "At threshold, a review modal opens automatically after /success."
        if auto_create
        else "At threshold, run /crystallize to open the review modal."
    )
    return (
        "Skill crystallization (quick guide):\n"
        "  Turn a repeated successful chat workflow into a reusable skill file.\n"
        "  1. Enable skills: /features skills on\n"
        "  2. Chat through a workflow (similar user messages each time)\n"
        f"  3. After each bot reply, run /success [optional note] "
        f"({min_repeats} times total; once per turn)\n"
        f"  4. {auto_line}\n"
        "  5. Review Trigger / Procedure / Validation, then Accept or Edit\n"
        "  6. Saved skills live in app/skills/active/ and inject when skills are on\n"
        f"  Fingerprint uses your last {window_turns} user messages. "
        "For the full walkthrough: /help crystallize"
    )


def format_crystallize_help_text(config: AppConfig | None = None) -> str:
    """Detailed help for /crystallize and the full skill crystallization workflow."""
    min_repeats, window_turns, auto_create = _skill_settings(config)
    auto_detail = (
        "When autoCreate is true in config.yaml, the app runs the crystallizer "
        "and opens the review modal automatically on the threshold /success."
        if auto_create
        else "When autoCreate is false (default), the threshold /success only "
        "reminds you to run /crystallize — nothing is drafted until you do."
    )
    return "\n".join(
        [
            "Skill crystallization — full guide",
            "",
            "What it does:",
            "  Repeatedly marking the same workflow as successful lets SoulForge",
            "  suggest a reusable skill (Trigger, Procedure, Validation) from your",
            "  chat. Nothing is saved until you Accept or run /skill-accept.",
            "",
            "Prerequisites:",
            "  • /features skills on  — required to save; marking /success works either way",
            "  • A repeatable workflow in chat (same kind of steps each time)",
            "",
            "Settings (config.yaml → skills:):",
            f"  • minSuccessfulRepeats: {min_repeats}  — /success marks needed before crystallize",
            f"  • successWindowTurns: {window_turns}  — user messages used for fingerprinting",
            f"  • autoCreate: {str(auto_create).lower()}  — {auto_detail}",
            "  • workflowLogPath — tracks counts in app/skills/workflow_log.json",
            "",
            "Step-by-step:",
            "  1. Chat normally. Example workflow:",
            "       You: I need to rebuild llama-cpp-python with CUDA in WSL",
            "       Bot:  (reply)",
            "       You: activate venv, export CUDA paths, build with GGML_CUDA",
            "       Bot:  (reply)",
            "",
            f"  2. Run /success [note] after a bot reply — e.g. /success rebuild cuda",
            "     Optional note becomes a hint for the skill name and summary.",
            "",
            "  3. Repeat similar chat + /success until the counter reaches "
            f"{min_repeats}/{min_repeats}.",
            "     • Each /success counts once per turn (same turn twice does not double-count).",
            "     • Use similar user phrasing so the fingerprint matches.",
            f"     • Only your last {window_turns} user messages are fingerprinted.",
            "",
            f"  4. When count reaches {min_repeats}:",
            "     • autoCreate true  → review modal opens after /success",
            "     • autoCreate false → system says to run /crystallize",
            "",
            "  5. /crystallize [fingerprint]",
            "     Drafts skill markdown from the logged workflow (LLM + validation).",
            "     Omit fingerprint to use the best eligible workflow from the log.",
            "",
            "  6. Review modal (TUI) or printed draft (CLI):",
            "     • Accept      — save to app/skills/active/<name>.md",
            "     • Edit        — change markdown, then save",
            "     • Reject      — discard draft (/skill-reject)",
            "",
            "  7. Verify: /skills lists the new skill; registry.json and workflow_log.json update.",
            "",
            "Related commands:",
            "  /success [note]     Mark workflow success (increments counter)",
            "  /crystallize        Draft pending skill from threshold workflow",
            "  /skill-accept       Save pending draft without modal",
            "  /skill-reject       Discard pending draft",
            "  /skills             Browse, create, or archive skills",
            "",
            "Tips:",
            "  • Lower minSuccessfulRepeats temporarily (e.g. 2) for faster testing.",
            "  • After Accept, the same workflow will not suggest again (crystallized_as in log).",
            "  • Saving requires skills on; enable before Accept if you marked success earlier.",
            "",
            "Back to all commands: /help",
        ]
    )


def format_sessions_help_text() -> str:
    return "\n".join(
        [
            "Session persistence — quick guide",
            "",
            "Saved conversations vs session.md:",
            "  • /session-save stores full chat history in app/sessions/<id>.json",
            "  • /session-summary writes a compact summary to app/memory/session.md",
            "  • session.md is injected when memory is enabled (short-term context)",
            "",
            "Workflow:",
            "  1. Chat normally",
            "  2. /session-save My project chat",
            "  3. Quit and restart later",
            "  4. /session-list then /session-load <id>",
            "  5. /session-summary before ending to capture notes in session.md",
            "",
            "Related commands:",
            "  /session-list      Browse saved sessions",
            "  /session-save    Persist current conversation",
            "  /session-load    Resume a saved conversation",
            "  /session-summary Generate and inject session summary",
            "",
            "Back to all commands: /help",
        ]
    )


def format_diagnostics_help_text() -> str:
    return "\n".join(
        [
            "Diagnostics — quick guide",
            "",
            "Commands:",
            "  /health       Short OK / warnings / errors with suggested fixes",
            "  /diagnostics  Full check list (models, CUDA, RAG, registry, kanban)",
            "  /config       Resolved paths and feature flags from config.yaml",
            "",
            "/status vs /health:",
            "  /status       Runtime snapshot (model, features, memory turns, RAG chunks)",
            "  /health       Setup validation before or after model load",
            "",
            "Startup:",
            "  Warnings appear in the welcome message and in logs/soulforge.log",
            "",
            "Back to all commands: /help",
        ]
    )


def format_tools_help_text() -> str:
    return "\n".join(
        [
            "Tool harness — quick guide",
            "",
            "Enable: /features tools on",
            "Configure: tools.allowWrite, tools.allowShell, readRoots in config.yaml",
            "",
            "Assistant proposes tools via ```tool JSON blocks in replies.",
            "Read-only tools may auto-run; write/shell/action tools need approval.",
            "",
            "Commands:",
            "  /tools           Open workshop (TUI) or status (CLI)",
            "  /tools test      Manual test: /tools test read_file '{\"path\":\"docs\"}'",
            "  /tools add-shell Add shellAllowlist entry",
            "  /tools allowlist List allowed shell prefixes",
            "  /tools-log       Audit log of tool events",
            "  /tool-approve    Execute a pending tool by ID",
            "  /tool-reject     Discard a pending tool",
            "",
            "Bridge tools (create_task, update_memory, create_skill) queue existing",
            "approval flows — they do not bypass /memory-accept or /skill-accept.",
            "",
            "Back to all commands: /help",
        ]
    )


def format_models_help_text() -> str:
    return "\n".join(
        [
            "Model routing - full guide",
            "",
            "What it does:",
            "  /models shows and edits model routing for both regular chat and",
            "  the opt-in multi-agent roles. /model still works for the existing",
            "  single chat model picker/import workflow.",
            "",
            "Commands:",
            "  /models",
            "    Show the current chat model, available ./models/*.gguf files,",
            "    each agent role's profile, effective model, and residency mode.",
            "",
            "  /models help",
            "    Show this guide.",
            "",
            "  /models chat <model>",
            "    Change the normal chat model. This is equivalent to /model <model>",
            "    and saves model.chatModelPath in config.yaml.",
            "",
            "  /models add <path> [switch]",
            "    Import a .gguf into ./models/. This is equivalent to /model add.",
            "",
            "  /models role <role> <model>",
            "    Assign one agent role to a specific model. SoulForge creates or",
            "    updates a role-specific profile named after that role, so changing",
            "    critic does not also change executor even if both previously shared",
            "    critic_executor. /models rule is accepted as an alias.",
            "",
            "  /models role <role> inherit",
            "    Keep the role-specific profile but clear its chatModelPath so the",
            "    role inherits the normal /model chat model.",
            "",
            "  /models profile <profile> <model>",
            "    Change a shared profile. Every role mapped to that profile is affected.",
            "    Use this for intentional shared changes like creator + synthesizer.",
            "",
            "  /models profile <profile> inherit",
            "    Clear a shared profile's chatModelPath so it follows /model.",
            "",
            "Valid roles:",
            "  orchestrator, researcher, creator, executor, critic, synthesizer",
            "",
            "Default profiles:",
            "  orchestrator    swap profile for high-level planning, often 70B",
            "  creator         resident profile for main construction, often 32B",
            "  critic_executor resident profile for review/tool work, often 8B",
            "",
            "Examples:",
            "  /models chat NemoMix",
            "  /models role orchestrator Qwen2.5-72B",
            "  /models role creator Qwen2.5-Coder-32B",
            "  /models role critic Llama-3.1-8B",
            "  /models role executor inherit",
            "  /models profile critic_executor Llama-3.1-8B",
            "",
            "Persistence:",
            "  Role/profile changes update agents.modelProfiles and agents.roles in",
            "  config.yaml. Loaded runtime profiles are unloaded so the next agent",
            "  task reloads the newly selected model.",
            "",
            "Back to all commands: /help",
        ]
    )


def format_agents_help_text() -> str:
    return "\n".join(
        [
            "Multi-agent workflows - full guide",
            "",
            "What it does:",
            "  /agents runs an opt-in local orchestration pipeline. A large",
            "  Orchestrator model creates a dependency-ordered strict-JSON task",
            "  graph, then worker roles execute scoped tasks and report back through",
            "  JSON envelopes. Normal chat is unchanged unless you start a run.",
            "",
            "Roles:",
            "  orchestrator  Plans and revises the task graph; usually the 70B swap profile.",
            "  researcher    Collects scoped local context from RAG/files/memory/skills.",
            "  creator       Builds the main deliverable; usually the resident 32B coder.",
            "  executor      Requests tools and summarizes execution logs.",
            "  critic        Validates outputs against the goal, schema, and criteria.",
            "  synthesizer   Produces the final user-facing answer after review passes.",
            "",
            "Commands:",
            "  /agents help",
            "    Show this detailed multi-agent workflow guide.",
            "",
            "  /agents",
            "    Show the active or latest run plus model profile residency. In TUI",
            "    this opens the Agent Workflow Status modal.",
            "",
            "  /agents on",
            "    Enable features.agents and save it to config.yaml. Agent runs remain",
            "    opt-in; regular chat still goes through the normal single-chat path.",
            "",
            "  /agents off",
            "    Disable agent runs. Existing run JSON files remain in app/agents/runs/.",
            "",
            "  /agents run <goal>",
            "    Start a new run. The Orchestrator emits a task graph, workers execute",
            "    dependency order, the Critic may request revisions, and the Synthesizer",
            "    returns final_answer when the run completes.",
            "",
            "  /agents status [run_id]",
            "    Inspect task ids, roles, status, attempts, dependencies, parent_task_id,",
            "    pending checkpoints, and final_answer. Omit run_id for active/latest.",
            "",
            "  /agents edit <task_id>",
            "    View a task's instructions, dependencies, parent, and current input_spec.",
            "    In TUI, opens an editor for the input_spec.",
            "",
            "  /agents edit <task_id> <new spec>",
            "    Replace the task input_spec manually when a role is stuck, using bad",
            "    context, or looping on a low-quality intermediate result.",
            "",
            "  /agents approve <checkpoint_id>",
            "    Execute a pending risky tool request through the existing ToolExecutor",
            "    permission system and audit log.",
            "",
            "  /agents reject <checkpoint_id>",
            "    Reject a pending tool request without executing it.",
            "",
            "  /agents cancel [run_id]",
            "    Mark the active or selected run as cancelled.",
            "",
            "State model:",
            "  Runs are stored as JSON under app/agents/runs/. Tasks can be pending,",
            "  running, passed, revising, paused, or blocked. Runs can be running,",
            "  paused, blocked, completed, or cancelled.",
            "",
            "Strict JSON protocol:",
            "  Every agent response must include schema_version, role, run_id, task_id,",
            "  parent_task_id, context_pruning, status, summary, artifacts,",
            "  tool_requests, next_actions, and errors. Invalid JSON gets one repair",
            "  attempt. A second schema failure blocks the task.",
            "",
            "Context pruning:",
            "  Worker prompts include the root goal, ancestor chain, dependencies, and",
            "  relevant artifacts. Unrelated sibling task chatter is excluded by default",
            "  through context_pruning.exclude_sibling_tasks.",
            "",
            "Tool safety:",
            "  Risky tool requests pause as checkpoints when agents.requireApproval is",
            "  true. Approval still obeys tools.allowWrite, tools.allowShell, readRoots,",
            "  writeRoots, shellAllowlist, and the existing tool audit log.",
            "",
            "Configuration:",
            "  agents.runsPath        Where run JSON files are stored.",
            "  agents.maxIterations   Revision limit before blocking.",
            "  agents.strictJson      Whether schema repair/blocking is enforced.",
            "  agents.residencyMode   hybrid or sequential model loading.",
            "  agents.modelProfiles   orchestrator, creator, critic_executor, etc.",
            "  agents.roles           Maps each role to a model profile.",
            "  /models                View and edit per-role model routing.",
            "  /models role <role> <model|inherit>",
            "                         Set one role's model without changing siblings.",
            "",
            "Runtime note:",
            "  The 70B Orchestrator profile may exceed 32GB VRAM and spill into",
            "  system memory, so planning can be slower than 32B/8B worker phases.",
            "  If resident 32B + 8B loading fails, SoulForge falls back to sequential",
            "  hot-swapping instead of failing startup.",
            "",
            "Typical flow:",
            "  1. /agents on",
            "  2. /agents run Build a small CLI utility and review it for edge cases",
            "  3. /agents status",
            "  4. /agents edit <task_id> <better context> if a task gets stuck",
            "  5. /agents approve <checkpoint_id> or /agents reject <checkpoint_id>",
            "",
            "Back to all commands: /help",
        ]
    )


def format_help_text(topic: str = "", config: AppConfig | None = None) -> str:
    """Format command help for /help or /help <topic>."""
    topic_key = topic.strip().lower()
    if topic_key in ("crystallize", "crystallization", "skills"):
        return format_crystallize_help_text(config)
    if topic_key in ("sessions", "session"):
        return format_sessions_help_text()
    if topic_key in ("diagnostics", "health", "config"):
        return format_diagnostics_help_text()
    if topic_key in ("tools", "tool"):
        return format_tools_help_text()
    if topic_key in ("agents", "agent"):
        return format_agents_help_text()
    if topic_key in ("models", "model", "model-routing", "routing"):
        return format_models_help_text()

    lines = ["Available commands:", ""]
    current_category = ""
    for cmd in COMMANDS:
        if cmd.category != current_category:
            current_category = cmd.category
            lines.append(f"{current_category}:")
        lines.append(f"  {cmd.usage}")
        lines.append(f"    {cmd.description}")
        lines.append("")

    lines.append(_skill_crystallization_summary(config))
    lines.append("")
    lines.append("Type anything else to chat.")
    return "\n".join(lines).rstrip()
