"""Shared command help registry for TUI and CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandHelp:
    category: str
    usage: str
    description: str


COMMANDS: tuple[CommandHelp, ...] = (
    CommandHelp("General", "/help", "Show available commands and what each does"),
    CommandHelp("General", "/status", "Show model name, active features, and app state"),
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
)


def format_help_text() -> str:
    """Format the full command list for /help."""
    lines = ["Available commands:", ""]
    current_category = ""
    for cmd in COMMANDS:
        if cmd.category != current_category:
            current_category = cmd.category
            lines.append(f"{current_category}:")
        lines.append(f"  {cmd.usage}")
        lines.append(f"    {cmd.description}")
        lines.append("")
    lines.append("Type anything else to chat.")
    return "\n".join(lines).rstrip()
