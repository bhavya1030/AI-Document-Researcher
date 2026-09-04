"""Persist embedded chunks in a local Qdrant collection.

This module does not parse PDFs, chunk text, or generate embeddings.
It only turns embedder output into Qdrant points (id + vector + payload).

Similarity search is intentionally not implemented here.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import QDRANT_COLLECTION_NAME, QDRANT_PATH

# Same chunk_id always maps to the same Qdrant point UUID (upsert, not duplicate).
_POINT_ID_NAMESPACE = uuid.UUID("8c0f2e1a-4b3d-4c9f-9a1e-7d6b5a4c3e2f")


class QdrantStoreError(Exception):
    """Raised when the local Qdrant store cannot complete an operation."""


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Deterministic UUID5 so a chunk_id is a stable Qdrant point id."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


class QdrantStore:
    """Local Qdrant wrapper: create collection, upsert batches, inspect counts."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        in_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self._vector_size = vector_size
        self.in_memory = in_memory
        self.path = Path(path) if path is not None else QDRANT_PATH

        try:
            if in_memory:
                self.client = QdrantClient(location=":memory:")
            else:
                self.path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(self.path))
        except Exception as exc:
            raise QdrantStoreError(f"Unable to initialize Qdrant client: {exc}") from exc

    def _resolve_vector_size(self) -> int:
        if self._vector_size is not None:
            if self._vector_size <= 0:
                raise QdrantStoreError("vector_size must be a positive integer")
            return self._vector_size

        # Imported lazily so unit tests can pass vector_size and skip the model.
        from app.ingestion.embedder import get_embedding_dimension

        self._vector_size = int(get_embedding_dimension())
        if self._vector_size <= 0:
            raise QdrantStoreError("Embedding model reported an invalid vector size")
        return self._vector_size

    def collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection_name))
        except Exception as exc:
            raise QdrantStoreError(
                f"Unable to check collection {self.collection_name!r}: {exc}"
            ) from exc

    def ensure_collection(self) -> None:
        """Create the collection if missing. Never drop an existing one."""
        vector_size = self._resolve_vector_size()

        if self.collection_exists():
            info = self.collection_info()
            if info["vector_size"] != vector_size:
                raise QdrantStoreError(
                    f"Collection {self.collection_name!r} already exists with "
                    f"vector size {info['vector_size']}, expected {vector_size}"
                )
            if info["distance"].lower() != "cosine":
                raise QdrantStoreError(
                    f"Collection {self.collection_name!r} already exists with "
                    f"distance {info['distance']}, expected Cosine"
                )
            return

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except Exception as exc:
            raise QdrantStoreError(
                f"Unable to create collection {self.collection_name!r}: {exc}"
            ) from exc

    def collection_info(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
        except Exception as exc:
            raise QdrantStoreError(
                f"Unable to read collection {self.collection_name!r}: {exc}"
            ) from exc

        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        points_count = getattr(info, "points_count", None)
        if points_count is None:
            points_count = self.count()

        return {
            "name": self.collection_name,
            "vector_size": int(size) if size is not None else None,
            "distance": getattr(distance, "value", str(distance)),
            "points_count": int(points_count or 0),
        }

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        try:
            result = self.client.count(collection_name=self.collection_name, exact=True)
        except Exception as exc:
            raise QdrantStoreError(
                f"Unable to count points in {self.collection_name!r}: {exc}"
            ) from exc
        return int(result.count)

    def upsert_chunks(self, embedded_chunks: list[dict[str, Any]]) -> int:
        """Insert or update embedder output. Empty input is a no-op."""
        if not embedded_chunks:
            return 0

        vector_size = self._resolve_vector_size()
        points: list[PointStruct] = []
        seen_ids: set[str] = set()

        for index, chunk in enumerate(embedded_chunks):
            point = self._point_from_chunk(chunk, index=index, vector_size=vector_size)
            if point.id in seen_ids:
                raise QdrantStoreError(
                    f"Duplicate chunk_id in batch: {chunk.get('chunk_id')!r}"
                )
            seen_ids.add(str(point.id))
            points.append(point)

        self.ensure_collection()

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            raise QdrantStoreError(f"Unable to upsert points: {exc}") from exc

        return len(points)

    def get_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Lookup one point by chunk_id. This is not similarity search."""
        if not chunk_id or not self.collection_exists():
            return None
        point_id = chunk_id_to_point_id(chunk_id)
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise QdrantStoreError(f"Unable to retrieve chunk {chunk_id!r}: {exc}") from exc

        if not records:
            return None
        record = records[0]
        payload = dict(record.payload or {})
        payload["id"] = str(record.id)
        return payload

    def _point_from_chunk(
        self,
        chunk: Any,
        *,
        index: int,
        vector_size: int,
    ) -> PointStruct:
        if not isinstance(chunk, dict):
            raise QdrantStoreError(f"Embedded chunk at index {index} is not a dict")

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise QdrantStoreError(f"Embedded chunk at index {index} is missing chunk_id")

        text = chunk.get("text")
        if not isinstance(text, str):
            raise QdrantStoreError(
                f"Embedded chunk {chunk_id!r} is missing text"
            )

        embedding = chunk.get("embedding")
        vector = _as_vector(embedding, chunk_id=chunk_id, expected_size=vector_size)

        metadata = chunk.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise QdrantStoreError(
                f"Embedded chunk {chunk_id!r} has invalid metadata"
            )

        filename = metadata.get("filename")
        page_number = metadata.get("page_number")
        if page_number is not None:
            try:
                page_number = int(page_number)
            except (TypeError, ValueError) as exc:
                raise QdrantStoreError(
                    f"Embedded chunk {chunk_id!r} has invalid page_number"
                ) from exc

        payload: dict[str, Any] = {
            "text": text,
            "filename": filename,
            "page_number": page_number,
            "chunk_id": chunk_id,
        }

        return PointStruct(
            id=chunk_id_to_point_id(chunk_id),
            vector=vector,
            payload=payload,
        )


def _as_vector(embedding: Any, *, chunk_id: str, expected_size: int) -> list[float]:
    if not isinstance(embedding, (list, tuple)):
        raise QdrantStoreError(
            f"Embedded chunk {chunk_id!r} is missing a numeric embedding"
        )
    try:
        vector = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise QdrantStoreError(
            f"Embedded chunk {chunk_id!r} has a non-numeric embedding"
        ) from exc
    if len(vector) != expected_size:
        raise QdrantStoreError(
            f"Embedded chunk {chunk_id!r} has dimension {len(vector)}, "
            f"expected {expected_size}"
        )
    return vector
