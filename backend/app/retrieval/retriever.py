"""Query → embedding → Qdrant nearest neighbors.

Uses the existing Embedder (same MiniLM model as ingestion) and QdrantStore.
Does not parse PDFs, chunk text, or generate answers.
"""

from __future__ import annotations

from typing import Any

from app.ingestion.embedder import get_embedder
from app.vectorstore.qdrant_store import QdrantStoreError, get_qdrant_store

DEFAULT_TOP_K = 5


class RetrievalError(Exception):
    """Raised when a query cannot be embedded or searched."""


class Retriever:
    """Embed a query with the shared model and search document_chunks."""

    def __init__(
        self,
        embedder: Any = None,
        store: Any = None,
    ) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def store(self) -> Any:
        if self._store is None:
            self._store = get_qdrant_store(vector_size=self.embedder.dimension)
        return self._store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top_k most similar chunks, highest cosine score first.

        Empty or whitespace-only queries return []. Missing collections and
        empty indexes also return [].
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if top_k <= 0:
            raise RetrievalError("top_k must be a positive integer")

        try:
            vector = self.embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(f"Unable to embed query: {exc}") from exc

        try:
            return self.store.search(
                vector,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except QdrantStoreError as exc:
            raise RetrievalError(f"Unable to search Qdrant: {exc}") from exc


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper around a default Retriever."""
    return Retriever().retrieve(query, top_k=top_k, score_threshold=score_threshold)
