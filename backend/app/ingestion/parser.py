"""PDF parsing: extract text page-by-page with page-level metadata."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError


class PdfParseError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


def parse_pdf(file_path: str | Path) -> list[dict]:
    """Extract text from a PDF, one entry per page.

    Returns a list of dictionaries. Each item has:
    - page_number: 1-based page index
    - text: extracted page text (empty string if none)
    - metadata: filename, page_number, extracted_text
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise PdfParseError(f"File not found: {path}")

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, PdfStreamError, OSError, ValueError) as exc:
        raise PdfParseError(f"Unable to parse PDF: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        decrypted = False
        try:
            decrypted = bool(reader.decrypt(""))
        except Exception as exc:
            raise PdfParseError("Encrypted PDFs are not supported") from exc
        if not decrypted and getattr(reader, "is_encrypted", False):
            raise PdfParseError("Encrypted PDFs are not supported")

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise PdfParseError(f"Unable to read PDF pages: {exc}") from exc

    if page_count == 0:
        raise PdfParseError("PDF has no pages")

    filename = path.name
    pages: list[dict] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text()
        except Exception:
            raw_text = None

        text = raw_text.strip() if isinstance(raw_text, str) else ""

        pages.append(
            {
                "page_number": index,
                "text": text,
                "metadata": {
                    "filename": filename,
                    "page_number": index,
                    "extracted_text": text,
                },
            }
        )

    return pages
