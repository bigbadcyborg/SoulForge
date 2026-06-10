"""CLI entry point for indexing docs/ into ChromaDB."""

from __future__ import annotations

from app.core.config import load_config
from app.core.model_runtime import ModelRuntime
from app.rag.ingest import ingest_documents


def main() -> None:
    config = load_config()
    runtime = ModelRuntime(config)

    def on_progress(name: str, method: str, current: int, total: int) -> None:
        print(f"[{current}/{total}] {name} ({method})")

    result = ingest_documents(config, runtime, on_progress=on_progress)
    print(result.summary())
    for line in result.skipped:
        print(f"  skipped: {line}")
    for line in result.errors:
        print(f"  note: {line}")


if __name__ == "__main__":
    main()
