"""Wire parsed pages through chunking, embedding, and Qdrant upsert.

Parser, chunker, embedder, and QdrantStore stay independent. This module
only calls them in order.
"""

from __future__ import annotations

from typing import Any

from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import get_embedder
from app.vectorstore.qdrant_store import get_qdrant_store


def ingest_pages(
    pages: list[dict[str, Any]],
    *,
    embedder: Any = None,
    store: Any = None,
) -> dict[str, int]:
    """Chunk → embed → upsert. Collection is created on first upsert if missing."""
    chunks = chunk_pages(pages)
    embedder = embedder or get_embedder()
    embedded = embedder.embed_chunks(chunks)
    store = store or get_qdrant_store(vector_size=getattr(embedder, "dimension", None))
    stored = store.upsert_chunks(embedded)
    return {
        "chunk_count": len(chunks),
        "vector_count": int(stored),
    }
