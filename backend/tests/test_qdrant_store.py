"""Independent tests for the Qdrant store (in-memory, no Docker, no embedder)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.vectorstore.qdrant_store import (
    QdrantStore,
    QdrantStoreError,
    chunk_id_to_point_id,
)

VECTOR_SIZE = 8


def _vector(fill: float) -> list[float]:
    return [fill + (index * 0.01) for index in range(VECTOR_SIZE)]


def _chunk(
    *,
    chunk_id: str,
    text: str,
    filename: str = "paper.pdf",
    page_number: int = 1,
    fill: float = 0.1,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "filename": filename,
            "page_number": page_number,
        },
        "embedding": _vector(fill),
    }


class QdrantStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantStore(in_memory=True, vector_size=VECTOR_SIZE)

    def test_client_initializes_successfully(self) -> None:
        self.assertIsNotNone(self.store.client)
        self.assertEqual(self.store.collection_name, "document_chunks")

    def test_collection_is_created_with_cosine_and_correct_dimension(self) -> None:
        self.assertFalse(self.store.collection_exists())
        self.store.ensure_collection()
        self.assertTrue(self.store.collection_exists())

        info = self.store.collection_info()
        self.assertEqual(info["vector_size"], VECTOR_SIZE)
        self.assertEqual(info["distance"].lower(), "cosine")
        self.assertEqual(info["points_count"], 0)

    def test_ensure_collection_is_idempotent(self) -> None:
        chunk = _chunk(chunk_id="paper_page_1_chunk_1", text="Keep this point.")
        self.store.upsert_chunks([chunk])
        self.assertEqual(self.store.count(), 1)

        self.store.ensure_collection()
        self.store.ensure_collection()

        self.assertTrue(self.store.collection_exists())
        self.assertEqual(self.store.count(), 1)
        info = self.store.collection_info()
        self.assertEqual(info["vector_size"], VECTOR_SIZE)
        self.assertEqual(info["distance"].lower(), "cosine")

    def test_single_chunk_is_inserted_with_payload(self) -> None:
        chunk = _chunk(
            chunk_id="paper_page_1_chunk_1",
            text="The Transformer architecture...",
            filename="paper.pdf",
            page_number=1,
        )
        inserted = self.store.upsert_chunks([chunk])

        self.assertEqual(inserted, 1)
        self.assertEqual(self.store.count(), 1)

        stored = self.store.get_by_chunk_id("paper_page_1_chunk_1")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["text"], "The Transformer architecture...")
        self.assertEqual(stored["filename"], "paper.pdf")
        self.assertEqual(stored["page_number"], 1)
        self.assertEqual(stored["chunk_id"], "paper_page_1_chunk_1")
        self.assertEqual(stored["id"], chunk_id_to_point_id("paper_page_1_chunk_1"))

    def test_multiple_chunks_are_inserted_in_a_batch(self) -> None:
        chunks = [
            _chunk(chunk_id="paper_page_1_chunk_1", text="First chunk.", fill=0.1),
            _chunk(
                chunk_id="paper_page_1_chunk_2",
                text="Second chunk.",
                page_number=1,
                fill=0.2,
            ),
            _chunk(
                chunk_id="paper_page_2_chunk_1",
                text="Third chunk.",
                page_number=2,
                fill=0.3,
            ),
        ]

        inserted = self.store.upsert_chunks(chunks)

        self.assertEqual(inserted, 3)
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(
            self.store.get_by_chunk_id("paper_page_2_chunk_1")["text"],
            "Third chunk.",
        )

    def test_stable_chunk_ids_upsert_instead_of_duplicating(self) -> None:
        first = _chunk(chunk_id="paper_page_1_chunk_1", text="Original text.", fill=0.1)
        updated = _chunk(chunk_id="paper_page_1_chunk_1", text="Updated text.", fill=0.4)

        self.store.upsert_chunks([first])
        self.store.upsert_chunks([updated])

        self.assertEqual(self.store.count(), 1)
        stored = self.store.get_by_chunk_id("paper_page_1_chunk_1")
        self.assertEqual(stored["text"], "Updated text.")
        self.assertEqual(stored["chunk_id"], "paper_page_1_chunk_1")

    def test_empty_batch_is_a_noop(self) -> None:
        self.assertEqual(self.store.upsert_chunks([]), 0)
        self.assertEqual(self.store.count(), 0)

    def test_wrong_vector_dimension_is_rejected(self) -> None:
        chunk = _chunk(chunk_id="paper_page_1_chunk_1", text="Bad vector.")
        chunk["embedding"] = [0.1, 0.2]

        with self.assertRaises(QdrantStoreError):
            self.store.upsert_chunks([chunk])
        self.assertEqual(self.store.count(), 0)

    def test_duplicate_chunk_ids_in_the_same_batch_are_rejected(self) -> None:
        chunks = [
            _chunk(chunk_id="paper_page_1_chunk_1", text="A", fill=0.1),
            _chunk(chunk_id="paper_page_1_chunk_1", text="B", fill=0.2),
        ]
        with self.assertRaises(QdrantStoreError):
            self.store.upsert_chunks(chunks)

    def test_malformed_chunks_are_rejected(self) -> None:
        with self.assertRaises(QdrantStoreError):
            self.store.upsert_chunks(["not-a-dict"])  # type: ignore[list-item]
        with self.assertRaises(QdrantStoreError):
            self.store.upsert_chunks([{"text": "missing ids and vector"}])
        with self.assertRaises(QdrantStoreError):
            self.store.upsert_chunks(
                [
                    {
                        "chunk_id": "paper_page_1_chunk_1",
                        "text": "no vector",
                        "metadata": {"filename": "paper.pdf", "page_number": 1},
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
