"""Model runtime: loads and serves the local GGUF chat and embedding models.

This module wraps ``llama_cpp.Llama`` so the rest of the app never touches the
backend directly. The heavy ``llama_cpp`` import is deferred until a model is
actually loaded, which keeps config/prompt logic importable in lightweight
environments (tests, tooling) without the native dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from app.core.config import AppConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_cpp import Llama


class ModelRuntime:
    """Owns the chat model and the optional embedding model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._chat: "Llama | None" = None
        self._embedder: "Llama | None" = None

    def load_chat_model(self) -> "Llama":
        """Load the GGUF chat model, raising a clear error if it is missing."""
        if self._chat is not None:
            return self._chat

        from llama_cpp import Llama

        model_path = self.config.model.chat_model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Chat model not found: {model_path}. "
                "Check model.chatModelPath in config.yaml."
            )

        print("Loading chat model...")
        self._chat = Llama(
            model_path=str(model_path),
            n_ctx=self.config.model.context_size,
            n_gpu_layers=self.config.model.gpu_layers,
            n_threads=self.config.model.threads,
            chat_format=self.config.model.chat_format,
            verbose=False,
        )
        return self._chat

    def load_embedding_model(self) -> "Llama":
        """Load the GGUF embedding model, raising a clear error if missing."""
        if self._embedder is not None:
            return self._embedder

        from llama_cpp import Llama

        model_path = self.config.model.embedding_model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found: {model_path}. "
                "Check model.embeddingModelPath in config.yaml, "
                "or disable RAG in config.yaml (features.rag: false)."
            )

        print("Loading embedding model...")
        self._embedder = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=2048,
            n_gpu_layers=self.config.model.gpu_layers,
            verbose=False,
        )
        return self._embedder

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a piece of text."""
        embedder = self.load_embedding_model()
        result = embedder.create_embedding(text)
        return result["data"][0]["embedding"]

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **overrides: Any,
    ) -> Any:
        """Run a chat completion using configured generation defaults.

        Any keyword in ``overrides`` takes precedence over the config values.
        Returns the raw llama_cpp response, or a streaming iterator when
        ``stream`` is True.
        """
        chat = self.load_chat_model()
        gen = self.config.generation

        params: dict[str, Any] = {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "repeat_penalty": gen.repeat_penalty,
            "max_tokens": gen.max_tokens,
            "stop": gen.stop,
        }
        params.update(overrides)

        return chat.create_chat_completion(
            messages=messages,
            stream=stream,
            **params,
        )

    @staticmethod
    def iter_stream_text(stream: Iterator[dict[str, Any]]) -> Iterator[str]:
        """Yield incremental text from a streaming chat completion."""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
