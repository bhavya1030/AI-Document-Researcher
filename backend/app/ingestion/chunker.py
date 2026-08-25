"""Split parsed PDF pages into retrieval-ready text chunks.

This module is independent of FastAPI routes, PDF parsing, embeddings,
and vector storage. It only turns page-level text into smaller chunks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Defaults used when callers do not pass their own values.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Leftover fragments shorter than this are merged into the previous chunk.
MIN_CHUNK_LENGTH = 20


def _clean_text(text: str) -> str:
    """Normalize whitespace without flattening useful paragraph breaks."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _filename_stem_for_id(filename: str) -> str:
    """Turn a filename into a stable, readable chunk-id prefix."""
    stem = Path(filename).stem
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return sanitized or "document"


def _page_filename(page: dict[str, Any]) -> str:
    metadata = page.get("metadata") or {}
    filename = metadata.get("filename")
    if isinstance(filename, str) and filename.strip():
        return Path(filename).name
    return "document.pdf"


def _page_number(page: dict[str, Any]) -> int:
    raw = page.get("page_number")
    if raw is None:
        metadata = page.get("metadata") or {}
        raw = metadata.get("page_number")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def _merge_tiny_chunks(pieces: list[str], min_chunk_length: int) -> list[str]:
    """Keep short leftover fragments from becoming their own chunks."""
    merged: list[str] = []
    for piece in pieces:
        text = piece.strip()
        if not text:
            continue
        if merged and len(text) < min_chunk_length:
            merged[-1] = f"{merged[-1]} {text}".strip()
        else:
            merged.append(text)
    return merged


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_length: int = MIN_CHUNK_LENGTH,
) -> list[dict[str, Any]]:
    """Convert parser page dicts into smaller chunks with preserved metadata.

    Each returned item looks like:
    {
        "chunk_id": "example_page_4_chunk_2",
        "text": "...",
        "metadata": {
            "filename": "example.pdf",
            "page_number": 4
        }
    }
    """
    splitter = _build_splitter(chunk_size, chunk_overlap)
    chunks: list[dict[str, Any]] = []

    for page in pages:
        raw_text = page.get("text") or ""
        if not isinstance(raw_text, str):
            continue

        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue

        filename = _page_filename(page)
        page_number = _page_number(page)
        id_stem = _filename_stem_for_id(filename)

        pieces = _merge_tiny_chunks(splitter.split_text(cleaned), min_chunk_length)

        for index, text in enumerate(pieces, start=1):
            chunks.append(
                {
                    "chunk_id": f"{id_stem}_page_{page_number}_chunk_{index}",
                    "text": text,
                    "metadata": {
                        "filename": filename,
                        "page_number": page_number,
                    },
                }
            )

    return chunks
