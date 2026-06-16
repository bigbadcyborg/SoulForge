"""Step definitions and copy for the first-launch tutorial wizard."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChecklistItem:
    """A single optional quest checkbox on a tutorial step."""

    id: str
    label: str


@dataclass(frozen=True)
class TutorialStep:
    """One screen in the onboarding wizard."""

    id: str
    title: str
    body: str
    checklist: tuple[ChecklistItem, ...] = field(default_factory=tuple)


TUTORIAL_STEPS: tuple[TutorialStep, ...] = (
    TutorialStep(
        id="welcome",
        title="Welcome to SoulForge",
        body=(
            "SoulForge is your local, uncensored AI workshop — persona, memory, "
            "documents, skills, tasks, and tools, all running on your machine.\n\n"
            "This quest walkthrough introduces the essentials. Nothing here changes "
            "your setup automatically; try commands in the chat when you're ready.\n\n"
            "Tips:\n"
            "  • Type /help for the full command list\n"
            "  • Commands start with / — regular text is sent to the model\n"
            "  • Ctrl+Q quits the TUI\n"
            "  • Re-open this wizard anytime with /tutorial"
        ),
        checklist=(
            ChecklistItem("read_welcome", "I understand how commands work"),
        ),
    ),
    TutorialStep(
        id="health",
        title="Quest 1 — Know Your Forge",
        body=(
            "Before diving in, learn how to check that everything is wired up.\n\n"
            "Try these commands:\n"
            "  /status   — model name, active features, compute backend\n"
            "  /health   — quick pass/warn/fail summary\n"
            "  /diagnostics — full system check (opens a modal)\n"
            "  /config   — resolved paths and limits (read-only)\n\n"
            "If something looks wrong, run ./doctor.sh from the project root "
            "for remediation hints."
        ),
        checklist=(
            ChecklistItem("tried_status", "Ran /status"),
            ChecklistItem("tried_health", "Ran /health or /diagnostics"),
        ),
    ),
    TutorialStep(
        id="persona",
        title="Quest 2 — Shape Your Persona",
        body=(
            "SoulForge injects SOUL.md into the system prompt when the soul "
            "feature is on (default).\n\n"
            "  • Edit SOUL.md at the project root to define tone and behavior\n"
            "  • Copy from examples/SOUL.example.md if you need a starter\n"
            "  • /reload-soul — reload persona after editing\n"
            "  • /features soul off — disable persona injection\n\n"
            "Your persona is the character of the forge — make it yours."
        ),
        checklist=(
            ChecklistItem("knows_soul", "I know where SOUL.md lives"),
            ChecklistItem("tried_reload", "Tried /reload-soul (optional)"),
        ),
    ),
    TutorialStep(
        id="rag",
        title="Quest 3 — Ground Answers in Your Docs",
        body=(
            "RAG retrieves relevant chunks from indexed documents in docs/.\n\n"
            "Setup flow:\n"
            "  1. Place .txt, .md, or .pdf files in docs/\n"
            "  2. /ingest — index them into the local vector store\n"
            "  3. /features rag on  (or /rag on)\n"
            "  4. Ask a question — use /sources to inspect retrieved chunks\n\n"
            "  /rag — open document selection modal (TUI)\n"
            "  /rag all — use every indexed document"
        ),
        checklist=(
            ChecklistItem("knows_docs", "I know docs/ is where files go"),
            ChecklistItem("tried_ingest", "Ran /ingest or plan to soon"),
            ChecklistItem("tried_rag", "Enabled RAG with /rag on"),
        ),
    ),
    TutorialStep(
        id="memory",
        title="Quest 4 — Persistent Memory",
        body=(
            "Memory keeps context across sessions in three markdown files:\n"
            "  • user.md    — facts about you\n"
            "  • memory.md  — long-term notes the bot should remember\n"
            "  • session.md — current-session scratch pad\n\n"
            "Commands:\n"
            "  /memory       — view all three sections\n"
            "  /memory-edit  — edit a section (e.g. /memory-edit user)\n"
            "  /memory-review — review AI-suggested memory updates\n"
            "  /memory-accept / /memory-reject — act on suggestions\n\n"
            "Every few turns the bot may propose memory updates for your approval."
        ),
        checklist=(
            ChecklistItem("viewed_memory", "Viewed /memory"),
            ChecklistItem("knows_edit", "I know how to edit memory files"),
        ),
    ),
    TutorialStep(
        id="skills",
        title="Quest 5 — Skills & Curator",
        body=(
            "Skills are reusable workflow recipes injected into prompts.\n\n"
            "  /features skills on  — enable skill injection\n"
            "  /skills              — browse active and archived skills\n"
            "  /success <note>      — mark a workflow step as successful\n"
            "  /crystallize         — turn repeated workflows into a skill\n"
            "  /help crystallize    — full crystallization guide\n\n"
            "Curator keeps skills healthy:\n"
            "  /features curator on\n"
            "  /curator-review      — review stale or bloated skills"
        ),
        checklist=(
            ChecklistItem("knows_skills", "Browsed /skills or know where they are"),
            ChecklistItem("knows_curator", "Know about /curator-review"),
        ),
    ),
    TutorialStep(
        id="kanban",
        title="Quest 6 — Kanban & Tasks",
        body=(
            "Track work on a local four-column board.\n\n"
            "  /features kanban on  — enable the task board\n"
            "  /tasks               — open the Kanban modal\n"
            "  /task-new <title>    — create a task from chat\n"
            "  /task-suggest        — let the bot propose task changes\n"
            "  /task-move, /task-done, /task-delete — manage tasks\n\n"
            "Tasks live in app/tasks/kanban.json — your forge's quest log."
        ),
        checklist=(
            ChecklistItem("enabled_kanban", "Enabled kanban or plan to"),
            ChecklistItem("opened_tasks", "Opened /tasks or created a task"),
        ),
    ),
    TutorialStep(
        id="tools",
        title="Quest 7 — Tools Workshop",
        body=(
            "Tools let the assistant read files, run shell commands, and more — "
            "with approval gates for risky operations.\n\n"
            "  /features tools on   — enable tool proposals in chat\n"
            "  /tools               — open the tools workshop (TUI)\n"
            "  /tools-log           — audit recent tool calls\n\n"
            "Safety defaults:\n"
            "  • Read-only tools may auto-approve\n"
            "  • Shell and write operations need your OK\n"
            "  • Configure allowlists in config.yaml under tools:\n\n"
            "Start with read-only tests before enabling shell or write."
        ),
        checklist=(
            ChecklistItem("knows_tools", "I know how to open /tools"),
            ChecklistItem("knows_approval", "I understand tool approval flow"),
        ),
    ),
    TutorialStep(
        id="finish",
        title="Forge Ready!",
        body=(
            "You're set to explore SoulForge on your own terms.\n\n"
            "Quick reference:\n"
            "  /help        — command catalog\n"
            "  /features    — toggle features (auto-saves)\n"
            "  /session-save / /session-load — save and restore chats\n"
            "  /tutorial    — re-open this wizard\n\n"
            "Happy forging! Ask anything, experiment with features, and shape "
            "the bot to fit your workflow."
        ),
        checklist=(
            ChecklistItem("ready", "I'm ready to chat!"),
        ),
    ),
)
