"""Document upload API routes."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.ingestion.parser import PdfParseError, parse_pdf

router = APIRouter()

# agentic-rag/data/documents  (resolved from this file's location)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

PDF_MAGIC = b"%PDF"


def _ensure_documents_dir() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _is_pdf_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".pdf"


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Accept a PDF, save it, parse it page-by-page, and return extracted text."""
    original_name = file.filename or ""
    safe_name = Path(original_name).name

    if not safe_name or not _is_pdf_filename(safe_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted. Upload a file with a .pdf extension.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if not content.lstrip().startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a valid PDF.",
        )

    _ensure_documents_dir()
    destination = DOCUMENTS_DIR / safe_name

    try:
        destination.write_bytes(content)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc

    try:
        parsed_pages = parse_pdf(destination)
    except PdfParseError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "filename": safe_name,
        "total_pages": len(parsed_pages),
        "pages": [
            {
                "page_number": page["page_number"],
                "text": page["text"],
            }
            for page in parsed_pages
        ],
    }
