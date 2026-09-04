"""Convert chunker output into dense embedding vectors.

This module is independent of FastAPI routes, PDF parsing, chunking,
and vector storage. It only turns chunk text into vectors.

The default model is sentence-transformers/all-MiniLM-L6-v2:
- runs locally (no API key)
- CPU-compatible
- 384-dimensional output, confirmed from the loaded model at runtime

Embeddings are L2-normalized so later cosine similarity in Qdrant can
be computed as a dot product. The vectors themselves stay in memory
for this step; nothing is written to a database.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


class Embedder:
    """Load an embedding model once and encode chunks in batches."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = SentenceTransformer(model_name)

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding width reported by the loaded model, not a hardcoded constant."""
        getter = getattr(self._model, "get_embedding_dimension", None)
        if not callable(getter):
            getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        dim = getter() if callable(getter) else None
        if not dim:
            raise RuntimeError(
                f"Could not read embedding dimension from model {self.model_name}"
            )
        return int(dim)

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach an embedding to each non-empty chunk.

        Empty input, missing text, and whitespace-only text are skipped so
        the model is never called on an empty string. Original chunk_id,
        text, and metadata are copied onto the result; input dicts are
        not mutated.
        """
        if not chunks:
            return []

        texts: list[str] = []
        kept: list[dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            texts.append(text)
            kept.append(chunk)

        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        dimension = self.dimension
        embedded: list[dict[str, Any]] = []
        for chunk, vector in zip(kept, vectors):
            metadata = chunk.get("metadata")
            embedding = [float(value) for value in vector]
            if len(embedding) != dimension:
                raise RuntimeError(
                    f"Model produced a vector of length {len(embedding)}, "
                    f"expected {dimension}"
                )
            embedded.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                    "embedding": embedding,
                }
            )

        return embedded


# One process-wide embedder so the model is not reloaded per request.
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the process-wide Embedder, creating it on first use."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_embedding_dimension() -> int:
    """Public dimension accessor for later Qdrant collection setup."""
    return get_embedder().dimension


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Encode chunks with the shared process-wide model."""
    return get_embedder().embed_chunks(chunks)
