"""Document text extraction for RAG ingestion (plain text + PDF/OCR)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import RagConfig

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".csv",
    ".html",
    ".css",
    ".js",
    ".pdf",
}


@dataclass
class LoadResult:
    text: str
    method: str  # "text" or "ocr"


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded."""


def check_ocr_available() -> tuple[bool, str]:
    """Return whether Tesseract and Poppler are available for PDF OCR."""
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        return False, "tesseract not found (install: sudo apt install tesseract-ocr)"

    try:
        subprocess.run(
            [tesseract, "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False, "tesseract not runnable"

    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        return False, "poppler not found (install: sudo apt install poppler-utils)"

    return True, ""


def _load_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as error:  # noqa: BLE001
        raise DocumentLoadError(f"Could not read PDF: {error}") from error

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as error:  # noqa: BLE001
            raise DocumentLoadError("Password-protected PDF") from error

    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _ocr_pdf(path: Path, lang: str, on_page=None) -> str:
    import pytesseract
    from pdf2image import convert_from_path

    try:
        images = convert_from_path(str(path))
    except Exception as error:  # noqa: BLE001
        raise DocumentLoadError(f"PDF to image conversion failed: {error}") from error

    pages: list[str] = []
    total = len(images)
    for index, image in enumerate(images, start=1):
        if on_page is not None:
            on_page(index, total)
        text = pytesseract.image_to_string(image, lang=lang)
        if text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages).strip()


def load_document_text(
    path: Path,
    config: RagConfig,
    *,
    on_ocr_page=None,
) -> LoadResult:
    """Load document text, using OCR for scanned PDFs when configured."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(f"Unsupported extension: {suffix}")

    if suffix != ".pdf":
        text = _load_plain_text(path)
        if not text.strip():
            raise DocumentLoadError("Empty file")
        return LoadResult(text=text.strip(), method="text")

    text = _extract_pdf_text(path)
    if len(text) >= config.pdf_min_text_chars:
        return LoadResult(text=text, method="text")

    if not config.pdf_ocr_enabled:
        if text.strip():
            return LoadResult(text=text, method="text")
        raise DocumentLoadError("Scanned PDF and OCR is disabled in config")

    ok, reason = check_ocr_available()
    if not ok:
        raise DocumentLoadError(f"OCR unavailable: {reason}")

    ocr_text = _ocr_pdf(path, config.pdf_ocr_lang, on_page=on_ocr_page)
    if not ocr_text.strip():
        raise DocumentLoadError("OCR produced no text")

    return LoadResult(text=ocr_text, method="ocr")
