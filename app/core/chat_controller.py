"""Chat controller: the non-UI heart of the chatbot.

Owns the model runtime, prompt assembly, memory, and RAG retrieval, and exposes
a small surface that both the CLI loop and the Textual TUI drive. Keeping this
logic UI-agnostic means the same code path powers every front end.
"""

from __future__ import annotations

from typing import Iterator

from app.core.config import PROJECT_ROOT, AppConfig
from app.core.feature_state import FeatureStateManager
from app.core.model_runtime import ModelRuntime
from app.core.prompt_builder import PromptBuilder
from app.memory.memory_manager import MemoryManager, MemorySnapshot
from app.rag.retriever import Retriever, RetrievedChunk

SOUL_PATH = PROJECT_ROOT / "SOUL.md"


def load_soul() -> str:
    """Load the persona from SOUL.md, or fall back to a neutral default."""
    if not SOUL_PATH.exists():
        return "You are a helpful local chatbot."
    return SOUL_PATH.read_text(encoding="utf-8", errors="ignore").strip()


class ChatController:
    """Coordinates models, prompts, memory, and retrieval for a session."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runtime = ModelRuntime(config)
        self.prompt_builder = PromptBuilder(config)
        self.memory_manager = MemoryManager(config)
        self.features = FeatureStateManager(config, on_change=self._on_feature_change)

        # RAG is initialized lazily to avoid loading embedding model at startup
        self._retriever: Retriever | None = None
        self.selected_sources: list[str] | None = None  # None = all sources

        self.soul_text: str = ""
        self.memory: MemorySnapshot | None = None
        self.messages: list[dict[str, str]] = []
        self.loaded: bool = False

    def _on_feature_change(self, key: str, enabled: bool) -> None:
        if key in ("soul", "memory"):
            self._rebuild_system_prompt()
        elif key == "rag" and enabled and self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

    def _initialize_retriever(self) -> Retriever | None:
        """Lazily initialize the retriever (only when RAG is first enabled)."""
        if self._retriever is not None:
            return self._retriever
        try:
            self._retriever = Retriever(self.config, self.runtime)
            self.runtime.load_embedding_model()
            return self._retriever
        except Exception as error:  # noqa: BLE001
            print(f"[rag] Failed to initialize retriever: {error}")
            return None

    @property
    def retriever(self) -> Retriever | None:
        """Return retriever if RAG is enabled, initializing if needed."""
        if self.features.is_enabled("rag"):
            return self._initialize_retriever()
        return None

    @property
    def rag_enabled(self) -> bool:
        """Whether RAG retrieval is active (compat for existing call sites)."""
        return self.features.is_enabled("rag")

    def load(self) -> None:
        """Load models and assemble the initial system prompt (blocking)."""
        self.soul_text = load_soul() if self.features.is_enabled("soul") else ""
        self.memory = (
            self.memory_manager.load() if self.features.is_enabled("memory") else None
        )

        self.runtime.load_chat_model()

        if self.features.is_enabled("rag") and self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

        self.messages = [{"role": "system", "content": self._build_system_prompt()}]
        self.loaded = True

    def _build_system_prompt(self) -> str:
        return self.prompt_builder.build_system_prompt(self.soul_text, self.memory)

    def _rebuild_system_prompt(self) -> None:
        """Reload soul/memory content and refresh the system message."""
        self.soul_text = load_soul() if self.features.is_enabled("soul") else ""
        self.memory = (
            self.memory_manager.load() if self.features.is_enabled("memory") else None
        )
        if self.messages:
            self.messages[0] = {
                "role": "system",
                "content": self._build_system_prompt(),
            }

    def reload_soul(self) -> None:
        """Reload SOUL.md and rebuild the system prompt without restarting."""
        self._rebuild_system_prompt()

    def add_user_turn(self, user_input: str) -> list[RetrievedChunk]:
        """Retrieve context, append the user message, and return any sources."""
        chunks: list[RetrievedChunk] = []
        context_text = ""
        if self.retriever is not None:
            chunks = self.retriever.retrieve(user_input)
            if self.selected_sources is not None and chunks:
                chunks = [c for c in chunks if c.source in self.selected_sources]
            context_text = Retriever.format_context(chunks)

        user_turn = self.prompt_builder.build_user_turn(
            user_input,
            context_text,
            use_rag=self.features.is_enabled("rag"),
        )
        self.messages.append({"role": "user", "content": user_turn})
        return chunks

    def toggle_rag(self) -> bool:
        """Toggle RAG on/off and return the new state."""
        return self.features.toggle("rag")

    def set_feature(self, key: str, enabled: bool) -> None:
        """Enable or disable a feature flag."""
        self.features.set_enabled(key, enabled)

    def apply_features(self, changes: dict[str, bool]) -> list[str]:
        """Apply multiple feature changes at once."""
        return self.features.apply_many(changes)

    def get_available_sources(self) -> list[str]:
        """Return available document sources from the vector store."""
        retriever = self._initialize_retriever()
        if retriever is None:
            return []
        return retriever.get_available_sources()

    def set_rag_sources(self, sources: list[str] | None) -> bool:
        """Set which document sources to use for RAG (None = all)."""
        if not self.features.is_enabled("rag"):
            return False
        self.selected_sources = sources
        return True

    def get_rag_status(self) -> dict[str, bool | list[str] | None]:
        """Return current RAG state, enabled status, and selected sources."""
        return {
            "enabled": self.features.is_enabled("rag"),
            "selected_sources": self.selected_sources,
            "available_sources": self.get_available_sources(),
        }

    def stream_reply(self) -> Iterator[str]:
        """Yield reply tokens, appending the full assistant message at the end."""
        stream = self.runtime.create_chat_completion(self.messages, stream=True)
        parts: list[str] = []
        for token in self.runtime.iter_stream_text(stream):
            parts.append(token)
            yield token
        self.messages.append(
            {"role": "assistant", "content": "".join(parts).strip()}
        )

    def full_reply(self) -> str:
        """Generate a complete (non-streamed) reply and append it."""
        response = self.runtime.create_chat_completion(self.messages, stream=False)
        reply = response["choices"][0]["message"]["content"].strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    @property
    def model_name(self) -> str:
        return self.config.model.chat_model.name

    def active_features(self) -> list[str]:
        return self.features.active_features()

    def features_summary(self) -> str:
        return self.features.summary()
