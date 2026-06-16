"""Quick smoke tests for iteration 6 memory nudging."""

from __future__ import annotations

from unittest.mock import patch

from app.core.chat_controller import ChatController
from app.core.config import load_config
from app.memory.memory_reviewer import (
    MemorySuggestion,
    _interpret_review_data,
    _parse_review_response,
    _resolve_section,
    build_local_suggestion,
    finalize_proposed_content,
    format_conversation_window,
    format_suggestion_view,
    likely_has_unsaved_facts,
)
from app.memory.memory_manager import MemorySnapshot


def test_parse_review_response() -> None:
    raw = (
        '{"action": "update", "section": "user", '
        '"rationale": "likes robots", "proposed_content": "User likes robots."}'
    )
    data = _parse_review_response(raw)
    assert data is not None
    assert data["section"] == "user"


def _review_context() -> tuple[MemorySnapshot, list[str]]:
    snapshot = MemorySnapshot(user="The user's favorite color is green.")
    user_statements = ["hello. im russ and i like robots", "what do you know about me?"]
    return snapshot, user_statements


def test_null_section_defaults_to_user() -> None:
    snapshot, user_statements = _review_context()
    data = {
        "action": "update",
        "section": None,
        "rationale": "new facts",
        "proposed_content": "Russ likes robots.",
    }
    assert _resolve_section(data, data["proposed_content"]) == "user"
    suggestion, error = _interpret_review_data(data, 10, "{}", snapshot, user_statements)
    assert error is None
    assert suggestion is not None
    assert suggestion.section == "user"


def test_missing_rationale_gets_default() -> None:
    snapshot, user_statements = _review_context()
    data = {
        "action": "update",
        "section": "user",
        "proposed_content": "Favorite color: green.",
    }
    suggestion, error = _interpret_review_data(data, 10, "{}", snapshot, user_statements)
    assert error is None
    assert suggestion is not None
    assert suggestion.rationale


def test_finalize_drops_hallucinated_lines() -> None:
    snapshot, user_statements = _review_context()
    proposed = (
        "The user's favorite color is green.\n"
        "Alice and Bob are friends. Alice checked in with Bob to see if he was free this weekend."
    )
    cleaned = finalize_proposed_content(proposed, snapshot, user_statements)
    assert "Alice" not in cleaned
    assert "Bob" not in cleaned
    assert "green" in cleaned.lower()
    assert "The user's name is Russ." in cleaned
    assert "The user likes robots." in cleaned
    assert "hello. im russ" not in cleaned.lower()
    assert "what do you know" not in cleaned.lower()


def test_finalize_summarizes_not_verbatim() -> None:
    snapshot = MemorySnapshot(user="The user's favorite color is green.")
    user_statements = ['"hello. im russ and i like robots"', '"what do you know about me?"']
    proposed = (
        "The user's favorite color is green.\n"
        '"hello. im russ and i like robots"\n'
        '"what do you know about me?"'
    )
    cleaned = finalize_proposed_content(proposed, snapshot, user_statements)
    assert cleaned == (
        "The user's favorite color is green.\n"
        "The user's name is Russ.\n"
        "The user likes robots."
    )


def test_model_echo_only_gets_enriched() -> None:
    snapshot, user_statements = _review_context()
    data = {
        "action": "update",
        "section": "user",
        "rationale": "already saved",
        "proposed_content": "The user's favorite color is green.",
    }
    suggestion, error = _interpret_review_data(data, 2, "{}", snapshot, user_statements)
    assert error is None
    assert suggestion is not None
    assert "The user's name is Russ." in suggestion.proposed_content
    assert "The user likes robots." in suggestion.proposed_content
    assert "hello. im russ" not in suggestion.proposed_content.lower()


def test_build_local_suggestion() -> None:
    snapshot, user_statements = _review_context()
    suggestion = build_local_suggestion(snapshot, user_statements, 2)
    assert suggestion is not None
    assert "Russ" in suggestion.proposed_content


def test_likely_has_unsaved_facts() -> None:
    snapshot = MemorySnapshot(user="The user's favorite color is green.")
    conversation = (
        "User: My name is Russ and I like robots.\n"
        "Assistant: noted.\n"
        "User: I work on SoulForge and use Python.\n"
        "Assistant: ok.\n"
        "User: I drink coffee and hate cilantro.\n"
        "Assistant: summary..."
    )
    assert likely_has_unsaved_facts(snapshot, conversation)


def test_format_conversation_window() -> None:
    msgs = [
        {"role": "user", "content": "ACTIVE MEMORY\n\nUSER MESSAGE:\nhello"},
        {"role": "assistant", "content": "hi there"},
    ]
    window = format_conversation_window(msgs, last_n_turns=10)
    assert "hello" in window
    assert "ACTIVE MEMORY" not in window


def test_turn_trigger_and_accept() -> None:
    controller = ChatController(load_config())
    controller.config.memory.update_every_turns = 3
    controller.turn_count = 3
    suggestion = MemorySuggestion("user", "test", "User likes robots.", 3)

    with patch(
        "app.core.chat_controller.generate_suggestion",
        return_value=(suggestion, None),
    ):
        assert controller.maybe_trigger_memory_review() is True
        assert controller.pending_suggestion is not None

    view = format_suggestion_view(controller.pending_suggestion, 3000)
    assert "turn 3" in view

    with patch.object(controller.memory_manager, "save", return_value=("saved", False)):
        with patch.object(controller, "reload_memory"):
            _, was_compacted = controller.accept_memory_suggestion()
    assert was_compacted is False
    assert controller.pending_suggestion is None


if __name__ == "__main__":
    test_parse_review_response()
    test_null_section_defaults_to_user()
    test_missing_rationale_gets_default()
    test_finalize_drops_hallucinated_lines()
    test_finalize_summarizes_not_verbatim()
    test_model_echo_only_gets_enriched()
    test_build_local_suggestion()
    test_likely_has_unsaved_facts()
    test_format_conversation_window()
    test_turn_trigger_and_accept()
    print("All memory nudge smoke tests passed.")
