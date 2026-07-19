"""Vector-backed episodic memory for past conversation turns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chromadb.api.models.Collection import Collection

EPISODIC_COLLECTION = "episodic_memory"


@dataclass
class EpisodicMemoryResult:
    id: str
    user: str
    assistant: str
    timestamp: str
    turn_count: int
    distance: float | None = None

    @property
    def preview(self) -> str:
        text = self.assistant or self.user
        return text[:240].rstrip()


class EpisodicMemoryStore:
    """Stores and searches conversation turns using the embedding model."""

    def __init__(self, config: AppConfig, runtime: ModelRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self._collection: "Collection | None" = None

    def _get_collection(self) -> "Collection | None":
        if self._collection is not None:
            return self._collection

        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.config.rag.db_dir))
            self._collection = client.get_or_create_collection(
                name=EPISODIC_COLLECTION
            )
        except Exception as error:  # noqa: BLE001
            print(f"[memory] Could not open episodic memory store: {error}")
            return None

        return self._collection

    def clear(self) -> int:
        """Delete all stored conversation turns from the episodic collection.

        Drops only the ``episodic_memory`` collection, leaving RAG documents in
        the same ChromaDB untouched. Returns the number of turns removed.
        """
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.config.rag.db_dir))
            try:
                existing = client.get_collection(name=EPISODIC_COLLECTION)
                count = existing.count()
            except Exception:  # noqa: BLE001 - collection may not exist yet
                count = 0
            client.delete_collection(name=EPISODIC_COLLECTION)
        except Exception as error:  # noqa: BLE001
            print(f"[memory] Could not clear episodic memory: {error}")
            return 0
        finally:
            self._collection = None
        return count

    def add_turn(
        self,
        *,
        turn_id: str,
        user: str,
        assistant: str,
        turn_count: int,
    ) -> bool:
        """Embed and store one user/assistant turn."""
        collection = self._get_collection()
        if collection is None:
            return False

        text = (
            f"User:\n{user.strip()}\n\n"
            f"Assistant:\n{assistant.strip()}"
        ).strip()
        if not text:
            return False

        try:
            embedding = self.runtime.embed(text)
            collection.upsert(
                ids=[turn_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[
                    {
                        "user": user.strip(),
                        "assistant": assistant.strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "turnCount": turn_count,
                    }
                ],
            )
            return True
        except Exception as error:  # noqa: BLE001
            print(f"[memory] Failed to store episodic memory: {error}")
            return False

    def search(self, query: str, *, limit: int = 5) -> list[EpisodicMemoryResult]:
        """Return relevant conversation episodes for a query."""
        collection = self._get_collection()
        if collection is None or not query.strip():
            return []

        try:
            embedding = self.runtime.embed(query)
            results = collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:  # noqa: BLE001
            print(f"[memory] Episodic search failed: {error}")
            return []

        ids = (results.get("ids") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        found: list[EpisodicMemoryResult] = []
        for index, item_id in enumerate(ids):
            metadata: dict[str, Any] = (
                metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            )
            distance = distances[index] if index < len(distances) else None
            found.append(
                EpisodicMemoryResult(
                    id=str(item_id),
                    user=str(metadata.get("user", "")),
                    assistant=str(metadata.get("assistant", "")),
                    timestamp=str(metadata.get("timestamp", "")),
                    turn_count=int(metadata.get("turnCount", 0) or 0),
                    distance=distance,
                )
            )
        return found

    def recent(self, *, limit: int = 10) -> list[EpisodicMemoryResult]:
        """Return recent stored turns when supported by the vector store."""
        collection = self._get_collection()
        if collection is None:
            return []
        try:
            results = collection.get(include=["metadatas"], limit=limit)
        except Exception as error:  # noqa: BLE001
            print(f"[memory] Failed to read episodic memory: {error}")
            return []

        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        found: list[EpisodicMemoryResult] = []
        for index, item_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            found.append(
                EpisodicMemoryResult(
                    id=str(item_id),
                    user=str(metadata.get("user", "")),
                    assistant=str(metadata.get("assistant", "")),
                    timestamp=str(metadata.get("timestamp", "")),
                    turn_count=int(metadata.get("turnCount", 0) or 0),
                )
            )
        return sorted(found, key=lambda item: item.timestamp, reverse=True)


def format_episodic_results(results: list[EpisodicMemoryResult]) -> str:
    """Format episodic search results for CLI/TUI display."""
    if not results:
        return "No episodic memory results."

    lines: list[str] = []
    for result in results:
        distance = (
            f", distance={result.distance:.4f}"
            if result.distance is not None
            else ""
        )
        lines.append(
            f"[{result.id}] turn {result.turn_count} ({result.timestamp}{distance})"
        )
        lines.append(f"User: {result.user or '(empty)'}")
        lines.append(f"Assistant: {result.assistant or '(empty)'}")
        lines.append("")
    return "\n".join(lines).rstrip()
