"""Prompt builder: assembles prompts in the order defined by plan section 10.

Assembly order (when every feature is enabled):
    1. Core system rules
    2. SOUL.md
    3. user.md
    4. memory.md
    5. session.md summary
    6. Relevant skills        (added in a later iteration)
    7. Retrieved RAG context
    8. Current user message

The builder respects feature toggles: disabled sections are simply skipped.
The system prompt carries rules + persona + memory; retrieved context is
attached to the user turn so the model treats it as grounding for the question.
"""

from __future__ import annotations

from app.core.config import AppConfig
from app.memory.memory_manager import MemorySnapshot

# Kept intentionally empty so the persona in SOUL.md remains the dominant voice.
# Future iterations can populate baseline operating rules here.
CORE_SYSTEM_RULES = ""

DEFAULT_SYSTEM_PROMPT = "You are a helpful local chatbot."


class PromptBuilder:
    """Builds system prompts and user turns from toggled feature inputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_system_prompt(
        self,
        soul_text: str = "",
        memory: MemorySnapshot | None = None,
    ) -> str:
        features = self.config.features
        sections: list[str] = []

        if CORE_SYSTEM_RULES.strip():
            sections.append(CORE_SYSTEM_RULES.strip())

        if features.soul and soul_text.strip():
            sections.append(soul_text.strip())

        if features.memory and memory is not None:
            if memory.user.strip():
                sections.append(f"# User\n{memory.user.strip()}")
            if memory.memory.strip():
                sections.append(f"# Memory\n{memory.memory.strip()}")
            if memory.session.strip():
                sections.append(f"# Session\n{memory.session.strip()}")

        if not sections:
            return DEFAULT_SYSTEM_PROMPT

        return "\n\n".join(sections)

    def build_user_turn(self, user_input: str, context_text: str = "") -> str:
        """Combine retrieved RAG context with the user's question."""
        if not (self.config.features.rag and context_text.strip()):
            return user_input

        return (
            "Use the retrieved context below when it is relevant.\n"
            "If the context does not contain the answer, say that the documents "
            "do not contain enough information and then answer from general "
            "knowledge if appropriate.\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"User question:\n{user_input}"
        )
