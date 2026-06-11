"""Memory reviewer: 10-turn nudging, suggestion generation, and compaction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime
from app.memory.memory_manager import SECTION_FILENAMES, MemoryManager, MemorySnapshot

REVIEW_SECTIONS = ("user", "memory")

REVIEW_COMPLETION_OVERRIDES = {
    "temperature": 0.2,
    "max_tokens": 1024,
    # Avoid chat stop tokens (User:/You:) truncating JSON mid-response.
    "stop": ["</s>"],
}

REVIEW_RETRY_PROMPT = (
    "Your previous response was invalid or incomplete. Reply with ONLY valid JSON.\n"
    'Required for updates: {"action": "update", "section": "user" or "memory", '
    '"rationale": "...", "proposed_content": "..."}\n'
    "section must be exactly user or memory (never null). "
    'If nothing to save: {"action": "none"}'
)

REVIEW_FORCE_UPDATE_PROMPT = (
    "You responded with action none, but user.md is nearly empty compared to what "
    "the user said in this conversation.\n\n"
    "IMPORTANT: Chat history is NOT saved. Only user.md and memory.md on disk persist.\n"
    "Extract durable facts from User: lines in the conversation (not Assistant replies).\n"
    "Merge with existing user.md content and respond with action update JSON.\n"
    "section must be \"user\" or \"memory\". proposed_content must be the full file body."
)

_STOPWORDS = frozenset(
    "a an the and or but is are was were be been being i you my me we they it its "
    "this that what do does did have has had can could would should will just also "
    "one more thing about from with for in on at to of not".split()
)


def _normalize_section(section: str) -> str | None:
    """Map model output to a valid review section key."""
    normalized = section.strip().lower().removesuffix(".md")
    if normalized in REVIEW_SECTIONS:
        return normalized
    return None


def _resolve_section(data: dict[str, Any], proposed: str) -> str:
    """Pick target file; default to user.md when section is missing or invalid."""
    raw = data.get("section")
    if raw is not None:
        text = str(raw).strip().lower()
        if text and text not in ("null", "none"):
            section = _normalize_section(str(raw))
            if section is not None:
                return section
    # Personal facts and preferences almost always belong in user.md.
    return "user"


@dataclass
class MemorySuggestion:
    section: str
    rationale: str
    proposed_content: str
    turn_count: int


def _strip_user_turn(content: str) -> str:
    """Extract the user's actual message from a wrapped user turn."""
    normalized = content.replace("\r\n", "\n")
    marker = "USER MESSAGE:\n"
    if marker in normalized:
        return normalized.split(marker, 1)[-1].strip()
    return normalized.strip()


def format_conversation_window(
    messages: list[dict[str, str]],
    last_n_turns: int = 10,
) -> str:
    """Format recent user/assistant exchanges for memory review."""
    pairs: list[str] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "").strip()
        if role == "user":
            pairs.append(f"User: {_strip_user_turn(content)}")
        elif role == "assistant":
            pairs.append(f"Assistant: {content}")

    if last_n_turns > 0:
        # Each turn is user + assistant; keep last 2 * N lines max
        pairs = pairs[-(last_n_turns * 2) :]

    return "\n".join(pairs) if pairs else "(no conversation yet)"


def _extract_user_messages(conversation: str) -> str:
    """Return combined text from User: lines in a formatted conversation window."""
    lines = [
        line[6:].strip()
        for line in conversation.splitlines()
        if line.startswith("User: ")
    ]
    return " ".join(lines)


def collect_user_statements(
    messages: list[dict[str, str]],
    last_n_turns: int = 10,
) -> list[str]:
    """Extract raw user messages from chat history for memory review."""
    statements: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        text = _strip_user_turn(message.get("content", "")).strip()
        if text:
            statements.append(text)
    if last_n_turns > 0:
        statements = statements[-last_n_turns:]
    return statements


def collect_user_statements_from_conversation(conversation: str) -> list[str]:
    """Fallback: pull user lines from a formatted conversation window."""
    statements = [
        line[6:].strip()
        for line in conversation.splitlines()
        if line.startswith("User: ")
    ]
    return [text for text in statements if text]


def _is_question(text: str) -> bool:
    stripped = text.strip().strip('"\'')
    if stripped.endswith("?"):
        return True
    return stripped.lower().startswith(
        ("what ", "who ", "where ", "when ", "why ", "how ", "do ", "does ", "did ", "can ")
    )


def _normalize_chat_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip('"\'').lower())


def _looks_like_raw_chat_line(line: str, user_statements: list[str]) -> bool:
    """True when a line is a verbatim or near-verbatim user message, not a memory note."""
    if _is_question(line):
        return True
    normalized = _normalize_chat_text(line)
    for statement in user_statements:
        stmt_norm = _normalize_chat_text(statement)
        if not stmt_norm:
            continue
        if normalized == stmt_norm:
            return True
        if stmt_norm in normalized or normalized in stmt_norm:
            if abs(len(normalized) - len(stmt_norm)) <= 5:
                return True
    return False


def _extract_declarative_facts(user_statements: list[str]) -> list[str]:
    """Turn declarative user chat into concise memory-style lines."""
    facts: list[str] = []

    def add(fact: str) -> None:
        if fact not in facts:
            facts.append(fact)

    for statement in user_statements:
        if _is_question(statement):
            continue
        text = statement.strip().strip('"\'')
        lower = text.lower()

        name_match = re.search(
            r"(?:^|[.\s,])(?:i'?m|i am|my name is)\s+([a-z][a-z'-]*)",
            lower,
        )
        if name_match:
            name = name_match.group(1).capitalize()
            add(f"The user's name is {name}.")

        like_match = re.search(r"i like (.+?)(?:[.!?]|$)", lower)
        if like_match:
            thing = like_match.group(1).strip().strip('"\'').rstrip(".")
            if thing:
                add(f"The user likes {thing}.")

        work_match = re.search(r"i work on (.+?)(?:[.!?]|$)", lower)
        if work_match:
            project = work_match.group(1).strip().rstrip(".")
            if project:
                add(f"The user works on {project}.")

    return facts


def _fact_already_covered(fact: str, corpus_lower: str) -> bool:
    lower = fact.lower()
    name_match = re.search(r"name is ([a-z][a-z'-]*)", lower)
    if name_match:
        return name_match.group(1) in corpus_lower
    like_match = re.search(r"likes (.+?)\.$", lower)
    if like_match:
        return like_match.group(1).strip() in corpus_lower
    work_match = re.search(r"works on (.+?)\.$", lower)
    if work_match:
        return work_match.group(1).strip() in corpus_lower
    return _line_supported(fact, corpus_lower)


def _merge_proposed_with_snapshot(proposed: str, snapshot: MemorySnapshot) -> str:
    """Use stored user.md as baseline, then layer any new model lines on top."""
    baseline = snapshot.user.strip()
    proposed = proposed.strip()
    if not baseline:
        return proposed
    if not proposed:
        return baseline
    lines = [line.strip() for line in baseline.splitlines() if line.strip()]
    for line in proposed.splitlines():
        stripped = line.strip()
        if stripped and stripped not in lines:
            lines.append(stripped)
    return "\n".join(lines)


def _allowed_corpus(snapshot: MemorySnapshot, user_statements: list[str]) -> str:
    parts = [snapshot.user, snapshot.memory, *user_statements]
    return "\n".join(part for part in parts if part).lower()


def _line_supported(line: str, allowed_lower: str) -> bool:
    line = line.strip()
    if not line:
        return False
    words = [
        w
        for w in re.findall(r"[a-z0-9']+", line.lower())
        if len(w) > 3 and w not in _STOPWORDS
    ]
    if not words:
        return True
    hits = sum(1 for word in words if word in allowed_lower)
    return hits >= max(1, len(words) * 2 // 3)


def finalize_proposed_content(
    proposed: str,
    snapshot: MemorySnapshot,
    user_statements: list[str],
) -> str:
    """Drop hallucinated or raw-chat lines; add concise facts from user statements."""
    allowed_lower = _allowed_corpus(snapshot, user_statements)
    kept: list[str] = []

    for line in proposed.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_raw_chat_line(stripped, user_statements):
            print(f"[memory] Dropped raw chat line from proposal: {stripped[:100]}")
            continue
        if _line_supported(stripped, allowed_lower):
            kept.append(stripped)
        else:
            print(f"[memory] Dropped unsupported proposed line: {stripped[:100]}")

    combined = "\n".join(kept).lower()
    for fact in _extract_declarative_facts(user_statements):
        if not _fact_already_covered(fact, combined):
            kept.append(fact)
            combined = f"{combined}\n{fact.lower()}"

    # Preserve existing user.md lines that were omitted by the model.
    if snapshot.user.strip():
        for line in snapshot.user.splitlines():
            stripped = line.strip()
            if stripped and stripped not in kept and _line_supported(stripped, allowed_lower):
                kept.insert(0, stripped)

    return "\n".join(kept).strip()


def build_local_suggestion(
    snapshot: MemorySnapshot,
    user_statements: list[str],
    turn_count: int,
) -> MemorySuggestion | None:
    """Build a suggestion from stored memory + extracted chat facts (no model)."""
    proposed = finalize_proposed_content(snapshot.user or "", snapshot, user_statements)
    if proposed.strip() == snapshot.user.strip():
        return None
    return MemorySuggestion(
        section="user",
        rationale="Suggested memory update from recent user statements.",
        proposed_content=proposed,
        turn_count=turn_count,
    )


def likely_has_unsaved_facts(snapshot: MemorySnapshot, conversation: str) -> bool:
    """Heuristic: user spoke durable facts not reflected in stored memory files."""
    stored = f"{snapshot.user}\n{snapshot.memory}".strip().lower()
    user_text = _extract_user_messages(conversation).strip()
    if len(user_text) < 30:
        return False

    stored_words = set(re.findall(r"[a-z0-9']+", stored))
    user_words = set(re.findall(r"[a-z0-9']+", user_text.lower()))
    novel = {w for w in user_words if w not in stored_words and w not in _STOPWORDS}

    if len(stored) < 100 and len(user_text) > 80:
        return True
    return len(novel) >= 4


def build_review_prompt(
    current_user: str,
    current_memory: str,
    user_statements: list[str],
) -> str:
    """Build the system prompt for memory review."""
    statements_block = (
        "\n".join(f"- {text}" for text in user_statements)
        if user_statements
        else "(no user statements)"
    )
    return (
        "You are a memory curator for a local chatbot. Review USER STATEMENTS and "
        "decide whether durable facts should be saved to memory files.\n\n"
        "Target files:\n"
        "- user.md: stable facts about the user (name, preferences, constraints)\n"
        "- memory.md: durable project or context memory\n\n"
        "Current user.md content:\n"
        f"{current_user or '(empty)'}\n\n"
        "Current memory.md content:\n"
        f"{current_memory or '(empty)'}\n\n"
        "USER STATEMENTS (ONLY source for NEW facts — do not invent anything else):\n"
        f"{statements_block}\n\n"
        "Respond with ONLY valid JSON in one of these forms:\n"
        '1. No update needed: {"action": "none"}\n'
        "2. Suggested update (section is REQUIRED — must be \"user\" or \"memory\", never null):\n"
        '{"action": "update", "section": "user", '
        '"rationale": "brief reason", '
        '"proposed_content": "full new file content including existing facts plus new ones"}\n\n'
        "Example update:\n"
        '{"action": "update", "section": "user", "rationale": "User said name Russ and likes robots", '
        '"proposed_content": "The user\'s name is Russ. Favorite color: green. Likes robots."}\n\n'
        "Rules:\n"
        "- NEVER invent people, names, or facts not present in USER STATEMENTS or current files.\n"
        "- Ignore Assistant messages entirely.\n"
        "- Skip questions; only save declarative facts the user stated about themselves.\n"
        "- proposed_content must merge current file content with new facts from USER STATEMENTS.\n"
        "- Write concise memory notes (e.g. \"The user's name is Russ.\"), never paste chat verbatim.\n"
        "- Use section \"user\" for personal facts; section \"memory\" for project/context notes.\n"
        '- Respond with action none ONLY if every declarative user fact is already in the files.'
    )


def _parse_review_response(raw: str) -> dict[str, Any] | None:
    """Parse JSON from the model response."""
    text = raw.strip()
    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    return data


def _interpret_review_data(
    data: dict[str, Any] | None,
    turn_count: int,
    raw: str,
    snapshot: MemorySnapshot,
    user_statements: list[str],
) -> tuple[MemorySuggestion | None, str | None]:
    """Convert parsed review JSON into a suggestion or error message."""
    if not data:
        preview = raw[:200].replace("\n", " ")
        message = f"Could not parse review response: {preview!r}"
        print(f"[memory] {message}\n[memory] Raw review response:\n{raw[:500]}")
        return None, message

    action = str(data.get("action", "")).strip().lower()
    proposed = str(data.get("proposed_content") or "").strip()
    rationale = str(data.get("rationale") or "").strip()

    if action == "none" or (action != "update" and not proposed):
        if action == "none" or not proposed:
            return None, None

    if not proposed:
        message = "Review response missing proposed_content."
        print(f"[memory] {message}\n[memory] Parsed review JSON: {data!r}")
        return None, message

    seed = _merge_proposed_with_snapshot(proposed, snapshot)
    proposed = finalize_proposed_content(seed, snapshot, user_statements)
    if proposed.strip() == snapshot.user.strip():
        local = build_local_suggestion(snapshot, user_statements, turn_count)
        if local is not None:
            proposed = local.proposed_content
        elif not _extract_declarative_facts(user_statements):
            return None, None
    if not proposed:
        message = "Review produced no supported memory content."
        print(f"[memory] {message}")
        return None, message

    if not rationale:
        rationale = "Suggested memory update from recent user statements."

    section = _resolve_section(data, proposed)
    return (
        MemorySuggestion(
            section=section,
            rationale=rationale,
            proposed_content=proposed,
            turn_count=turn_count,
        ),
        None,
    )


def _run_review_completion(
    runtime: ModelRuntime,
    review_messages: list[dict[str, str]],
) -> str:
    response = runtime.create_chat_completion(
        review_messages,
        stream=False,
        **REVIEW_COMPLETION_OVERRIDES,
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_suggestion(
    runtime: ModelRuntime,
    config: AppConfig,
    messages: list[dict[str, str]],
    snapshot: MemorySnapshot,
    turn_count: int,
) -> tuple[MemorySuggestion | None, str | None]:
    """Run a one-shot review. Returns (suggestion, error_message)."""
    interval = config.memory.update_every_turns
    window_turns = interval if interval > 0 else 10
    conversation = format_conversation_window(messages, last_n_turns=window_turns)
    user_statements = collect_user_statements(messages, last_n_turns=window_turns)
    if not user_statements:
        user_statements = collect_user_statements_from_conversation(conversation)
    system_prompt = build_review_prompt(
        snapshot.user,
        snapshot.memory,
        user_statements,
    )

    review_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Analyze the conversation and respond with JSON only."},
    ]

    try:
        raw = _run_review_completion(runtime, review_messages)
    except Exception as error:  # noqa: BLE001
        message = f"Review generation failed: {error}"
        print(f"[memory] {message}")
        return None, message

    suggestion, error = _interpret_review_data(
        _parse_review_response(raw),
        turn_count,
        raw,
        snapshot,
        user_statements,
    )
    if suggestion is not None:
        return suggestion, None
    if error is not None:
        print(f"[memory] Review retry after: {error}")
        retry_messages = [
            *review_messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": REVIEW_RETRY_PROMPT},
        ]
        try:
            raw_retry = _run_review_completion(runtime, retry_messages)
        except Exception as retry_error:  # noqa: BLE001
            message = f"Review retry failed: {retry_error}"
            print(f"[memory] {message}")
            return None, message

        suggestion, error = _interpret_review_data(
            _parse_review_response(raw_retry),
            turn_count,
            raw_retry,
            snapshot,
            user_statements,
        )
        if suggestion is not None:
            return suggestion, None

    if likely_has_unsaved_facts(snapshot, conversation):
        print(
            "[memory] Review returned none but conversation has unsaved facts; "
            "forcing update retry."
        )
        force_messages = [
            *review_messages,
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"{REVIEW_FORCE_UPDATE_PROMPT}\n\n"
                    f"Current user.md:\n{snapshot.user or '(empty)'}\n\n"
                    f"Current memory.md:\n{snapshot.memory or '(empty)'}"
                ),
            },
        ]
        try:
            raw_force = _run_review_completion(runtime, force_messages)
        except Exception as force_error:  # noqa: BLE001
            message = f"Review force-update failed: {force_error}"
            print(f"[memory] {message}")
            return None, message

        forced, force_err = _interpret_review_data(
            _parse_review_response(raw_force),
            turn_count,
            raw_force,
            snapshot,
            user_statements,
        )
        if forced is not None:
            return forced, None
        if force_err is not None:
            return None, force_err

    local = build_local_suggestion(snapshot, user_statements, turn_count)
    if local is not None:
        print("[memory] Using locally extracted facts for memory suggestion.")
        return local, None

    return None, None


def compact_content(
    runtime: ModelRuntime,
    section: str,
    content: str,
    max_chars: int,
) -> tuple[str, bool]:
    """Summarize memory content to fit within max_chars using the LLM."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content.strip(), False

    filename = SECTION_FILENAMES.get(section, f"{section}.md")
    system_prompt = (
        f"Compress the following {filename} content to at most {max_chars} characters. "
        "Preserve all important facts. Output only the compressed text, no commentary."
    )
    compact_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    try:
        response = runtime.create_chat_completion(
            compact_messages,
            stream=False,
            **REVIEW_COMPLETION_OVERRIDES,
        )
        compacted = response["choices"][0]["message"]["content"].strip()
    except Exception as error:  # noqa: BLE001
        print(f"[memory] Compaction failed: {error}")
        compacted = content[:max_chars].rstrip()
        return compacted, True

    if len(compacted) > max_chars:
        compacted = compacted[:max_chars].rstrip()
    return compacted, True


def merge_and_compact(
    runtime: ModelRuntime,
    section: str,
    existing: str,
    proposed: str,
    max_chars: int,
) -> tuple[str, bool]:
    """Ensure proposed content fits; compact via LLM if needed."""
    text = MemoryManager._strip_html_comments(proposed.strip())
    if not text:
        text = proposed.strip()
    return compact_content(runtime, section, text, max_chars)


def format_suggestion_view(
    suggestion: MemorySuggestion,
    max_chars: int,
) -> str:
    """Format a pending suggestion for display."""
    filename = SECTION_FILENAMES.get(suggestion.section, f"{suggestion.section}.md")
    count = len(suggestion.proposed_content)
    lines = [
        f"Memory review (turn {suggestion.turn_count})",
        f"Target: {filename} ({count}/{max_chars} chars)",
        "",
        "Rationale:",
        suggestion.rationale,
        "",
        "Proposed content:",
        suggestion.proposed_content or "(empty)",
        "",
        "Use /memory-accept to save, /memory-reject to discard, or Edit in TUI.",
    ]
    return "\n".join(lines)
