"""RAG document ingestion: docs/ → ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime
from app.rag.chunker import chunk_text
from app.rag.loaders import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    check_ocr_available,
    load_document_text,
)

ProgressCallback = Callable[[str, str, int, int], None]
# args: filename, method, current_index, total_files


@dataclass
class IngestResult:
    files_processed: int = 0
    chunks_indexed: int = 0
    ocr_files: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Indexed {self.chunks_indexed} chunk(s) from {self.files_processed} file(s)."
        ]
        if self.ocr_files:
            lines.append(f"OCR used for {self.ocr_files} PDF(s).")
        if self.skipped:
            lines.append(f"Skipped {len(self.skipped)} file(s).")
        return "\n".join(lines)


def discover_documents(docs_dir: Path) -> list[Path]:
    """Return supported files under docs_dir (non-recursive)."""
    if not docs_dir.exists():
        return []
    files = [
        path
        for path in sorted(docs_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def _source_name(path: Path, docs_dir: Path) -> str:
    try:
        return path.relative_to(docs_dir).as_posix()
    except ValueError:
        return path.name


def ingest_documents(
    config: AppConfig,
    runtime: ModelRuntime,
    *,
    on_progress: ProgressCallback | None = None,
) -> IngestResult:
    """Index all supported documents from docs/ into ChromaDB."""
    result = IngestResult()
    docs_dir = config.rag.docs_dir
    files = discover_documents(docs_dir)

    if not files:
        result.errors.append(f"No supported documents found in {docs_dir}")
        return result

    pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
    if pdf_files and config.rag.pdf_ocr_enabled:
        ok, reason = check_ocr_available()
        if not ok:
            result.errors.append(
                f"PDF OCR deps missing ({reason}). "
                "Digital PDFs may still work; scanned PDFs will be skipped."
            )

    try:
        import chromadb
    except ImportError as error:
        result.errors.append(f"chromadb not installed: {error}")
        return result

    runtime.load_embedding_model()

    client = chromadb.PersistentClient(path=str(config.rag.db_dir))
    collection = client.get_or_create_collection(name=config.rag.collection_name)

    total = len(files)
    for index, path in enumerate(files, start=1):
        source = _source_name(path, docs_dir)

        def ocr_page(page_num: int, page_total: int) -> None:
            if on_progress is not None:
                on_progress(source, f"ocr page {page_num}/{page_total}", index, total)

        try:
            if on_progress is not None:
                on_progress(source, "loading", index, total)

            loaded = load_document_text(
                path,
                config.rag,
                on_ocr_page=ocr_page if path.suffix.lower() == ".pdf" else None,
            )
        except DocumentLoadError as error:
            result.skipped.append(f"{source}: {error}")
            continue
        except Exception as error:  # noqa: BLE001
            result.skipped.append(f"{source}: {error}")
            continue

        pieces = chunk_text(
            loaded.text,
            config.rag.chunk_size,
            config.rag.chunk_overlap,
        )
        if not pieces:
            result.skipped.append(f"{source}: no chunks produced")
            continue

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk_index, piece in enumerate(pieces):
            if on_progress is not None:
                on_progress(
                    source,
                    f"{loaded.method} chunk {chunk_index + 1}/{len(pieces)}",
                    index,
                    total,
                )
            chunk_id = f"{source}::{chunk_index}"
            ids.append(chunk_id)
            embeddings.append(runtime.embed(piece))
            documents.append(piece)
            metadatas.append(
                {
                    "source": source,
                    "chunkIndex": chunk_index,
                    "extractionMethod": loaded.method,
                }
            )

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as error:  # noqa: BLE001
            result.skipped.append(f"{source}: upsert failed ({error})")
            continue

        result.files_processed += 1
        result.chunks_indexed += len(pieces)
        if loaded.method == "ocr":
            result.ocr_files += 1

    return result
