"""Model runtime: loads and serves the local GGUF chat and embedding models.

This module wraps ``llama_cpp.Llama`` so the rest of the app never touches the
backend directly. The heavy ``llama_cpp`` import is deferred until a model is
actually loaded, which keeps config/prompt logic importable in lightweight
environments (tests, tooling) without the native dependency.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from app.core.compute_backend import UNKNOWN, ComputeBackend, detect_compute_backend
from app.core.config import AgentModelProfileConfig, AppConfig
from app.utils.audit_logger import append_audit_event

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_cpp import Llama


@dataclass
class RuntimeProfileStatus:
    name: str
    loaded: bool
    residency: str
    model_path: str


class ModelRuntime:
    """Owns the chat model and the optional embedding model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._chat: "Llama | None" = None
        self._chat_profiles: dict[str, "Llama"] = {}
        self._profile_paths: dict[str, Path] = {}
        self._embedder: "Llama | None" = None
        self._compute_backend: ComputeBackend = UNKNOWN
        self._lock = threading.RLock()
        self._active_profile: str = ""
        self._resident_fallback: bool = False
        self._profile_errors: list[str] = []

    @property
    def compute_backend(self) -> ComputeBackend:
        return self._compute_backend

    def load_chat_model(self) -> "Llama":
        """Load the GGUF chat model, raising a clear error if it is missing."""
        with self._lock:
            chat = self._load_chat_profile_unlocked("default")
            self._chat = chat
            return chat

    def unload_chat_model(self) -> None:
        """Release all loaded chat profiles and reset compute backend detection."""
        with self._lock:
            self._chat = None
            self._chat_profiles.clear()
            self._profile_paths.clear()
            self._active_profile = ""
            self._compute_backend = UNKNOWN
        gc.collect()

    def reload_chat_model(self) -> "Llama":
        """Unload and load the chat model from the current config path."""
        with self._lock:
            self._chat = None
            self._chat_profiles.clear()
            self._profile_paths.clear()
            self._active_profile = ""
            self._compute_backend = UNKNOWN
        gc.collect()
        with self._lock:
            chat = self._load_chat_profile_unlocked("default", force=True)
            self._chat = chat
            return chat

    def unload_chat_profile(self, profile_name: str) -> None:
        """Release a single named chat profile if it is loaded."""
        key = self._normalize_profile_name(profile_name)
        with self._lock:
            self._chat_profiles.pop(key, None)
            self._profile_paths.pop(key, None)
            if key == "default":
                self._chat = None
            if self._active_profile == key:
                self._active_profile = ""
        gc.collect()

    def warm_resident_profiles(self) -> list[str]:
        """Best-effort load of resident agent profiles.

        Returns warning strings. Resident loading is opportunistic by design:
        if the local VRAM budget cannot hold the configured resident models, the
        runtime unloads partial state and falls back to sequential hot-swapping.
        """
        warnings: list[str] = []
        if self.config.agents.residency_mode != "hybrid":
            self._resident_fallback = True
            return warnings

        resident_names = [
            name
            for name, profile in self.config.agents.model_profiles.items()
            if profile.residency == "resident"
        ]
        for name in resident_names:
            try:
                with self._lock:
                    self._load_chat_profile_unlocked(name)
            except Exception as error:  # noqa: BLE001
                warnings.append(
                    f"Resident profile '{name}' failed to load; using sequential fallback: {error}"
                )
                with self._lock:
                    self._chat_profiles.clear()
                    self._profile_paths.clear()
                    self._chat = None
                    self._resident_fallback = True
                    self._profile_errors.extend(warnings)
                gc.collect()
                break
        return warnings

    def profile_statuses(self) -> list[RuntimeProfileStatus]:
        """Return loaded/unloaded state for configured runtime profiles."""
        statuses = [
            RuntimeProfileStatus(
                name="default",
                loaded="default" in self._chat_profiles,
                residency="resident",
                model_path=str(self.config.model.chat_model),
            )
        ]
        for name, profile in self.config.agents.model_profiles.items():
            statuses.append(
                RuntimeProfileStatus(
                    name=name,
                    loaded=name in self._chat_profiles,
                    residency=profile.residency,
                    model_path=str(self._profile_model_path(name)),
                )
            )
        return statuses

    def load_embedding_model(self) -> "Llama":
        """Load the GGUF embedding model, raising a clear error if missing."""
        with self._lock:
            return self._load_embedding_model_unlocked()

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a piece of text."""
        with self._lock:
            embedder = self._load_embedding_model_unlocked()
            result = embedder.create_embedding(text)
            return result["data"][0]["embedding"]

    def _completion_params(self, **overrides: Any) -> dict[str, Any]:
        gen = self.config.generation
        params: dict[str, Any] = {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "repeat_penalty": gen.repeat_penalty,
            "max_tokens": gen.max_tokens,
            "stop": gen.stop,
        }
        params.update(overrides)
        return params

    def _normalize_profile_name(self, profile_name: str | None) -> str:
        key = (profile_name or "default").strip()
        return key or "default"

    def _profile_config(self, profile_name: str) -> AgentModelProfileConfig | None:
        if profile_name == "default":
            return None
        return self.config.agents.model_profiles.get(profile_name)

    def _profile_model_path(self, profile_name: str) -> Path:
        profile = self._profile_config(profile_name)
        if profile is not None and profile.chat_model is not None:
            return profile.chat_model
        return self.config.model.chat_model

    def _profile_residency(self, profile_name: str) -> str:
        profile = self._profile_config(profile_name)
        if profile is None:
            return "resident"
        return profile.residency

    def _profile_completion_overrides(self, profile_name: str) -> dict[str, Any]:
        profile = self._profile_config(profile_name)
        if profile is None:
            return {}
        overrides: dict[str, Any] = {}
        if profile.temperature is not None:
            overrides["temperature"] = profile.temperature
        if profile.top_p is not None:
            overrides["top_p"] = profile.top_p
        if profile.repeat_penalty is not None:
            overrides["repeat_penalty"] = profile.repeat_penalty
        if profile.max_tokens is not None:
            overrides["max_tokens"] = profile.max_tokens
        return overrides

    def _load_chat_profile_unlocked(
        self,
        profile_name: str,
        *,
        force: bool = False,
    ) -> "Llama":
        key = self._normalize_profile_name(profile_name)
        model_path = self._profile_model_path(key)
        loaded = self._chat_profiles.get(key)
        if loaded is not None and not force and self._profile_paths.get(key) == model_path:
            self._active_profile = key
            return loaded

        residency = self._profile_residency(key)
        if residency not in ("resident", "swap"):
            raise ValueError(
                f"Invalid residency for profile '{key}': {residency}. "
                "Use 'resident' or 'swap'."
            )

        if force:
            self._chat_profiles.pop(key, None)
            self._profile_paths.pop(key, None)

        if residency == "swap" or self._resident_fallback or self.config.agents.residency_mode == "sequential":
            for loaded_key in list(self._chat_profiles):
                if loaded_key != key:
                    self._chat_profiles.pop(loaded_key, None)
                    self._profile_paths.pop(loaded_key, None)
            self._chat = None
            gc.collect()

        from llama_cpp import Llama

        if not model_path.exists():
            raise FileNotFoundError(
                f"Chat model not found for profile '{key}': {model_path}. "
                "Check model.chatModelPath or agents.modelProfiles in config.yaml."
            )

        print(f"Loading chat model profile '{key}'...")
        profile = self._profile_config(key)
        chat_format = (
            profile.chat_format
            if profile is not None and profile.chat_format
            else self.config.model.chat_format
        )
        chat = Llama(
            model_path=str(model_path),
            n_ctx=self.config.model.context_size,
            n_gpu_layers=self.config.model.gpu_layers,
            n_threads=self.config.model.threads,
            chat_format=chat_format,
            verbose=False,
        )
        self._chat_profiles[key] = chat
        self._profile_paths[key] = model_path
        self._active_profile = key
        if key == "default":
            self._chat = chat
        self._compute_backend = detect_compute_backend(self.config)
        return chat

    def _load_embedding_model_unlocked(self) -> "Llama":
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
        return self.create_chat_completion_for_profile(
            "default",
            messages,
            stream=stream,
            **overrides,
        )

    def create_chat_completion_for_profile(
        self,
        profile_name: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        **overrides: Any,
    ) -> Any:
        """Run a chat completion against a named runtime profile."""
        profile_overrides = self._profile_completion_overrides(profile_name)
        profile_overrides.update(overrides)
        params = self._completion_params(**profile_overrides)
        if stream:
            return self._stream_chat_completion_for_profile(
                profile_name,
                messages,
                **profile_overrides,
            )

        with self._lock:
            chat = self._load_chat_profile_unlocked(profile_name)
            try:
                response = chat.create_chat_completion(
                    messages=messages,
                    stream=False,
                    **params,
                )
            except Exception as error:
                append_audit_event(
                    self.config,
                    messages=messages,
                    params=params,
                    stream=False,
                    error=str(error),
                )
                raise

            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            append_audit_event(
                self.config,
                messages=messages,
                params=params,
                response_text=text,
                stream=False,
            )
            return response

    def _stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> Iterator[dict[str, Any]]:
        return self._stream_chat_completion_for_profile(
            "default",
            messages,
            **overrides,
        )

    def _stream_chat_completion_for_profile(
        self,
        profile_name: str,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream tokens while holding the runtime lock for the full decode."""
        params = self._completion_params(**overrides)
        with self._lock:
            chat = self._load_chat_profile_unlocked(profile_name)
            response_parts: list[str] = []
            try:
                stream = chat.create_chat_completion(
                    messages=messages,
                    stream=True,
                    **params,
                )
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        response_parts.append(content)
                    yield chunk
            except Exception as error:
                append_audit_event(
                    self.config,
                    messages=messages,
                    params=params,
                    response_text="".join(response_parts),
                    stream=True,
                    error=str(error),
                )
                raise
            append_audit_event(
                self.config,
                messages=messages,
                params=params,
                response_text="".join(response_parts),
                stream=True,
            )

    @staticmethod
    def iter_stream_text(stream: Iterator[dict[str, Any]]) -> Iterator[str]:
        """Yield incremental text from a streaming chat completion."""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
