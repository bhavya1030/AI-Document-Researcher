"""Semantic retrieval tests (in-memory Qdrant; no Docker, no FastAPI)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.embedder import get_embedder
from app.retrieval.retriever import RetrievalError, Retriever
from app.vectorstore.qdrant_store import QdrantStore

VECTOR_SIZE = 8


class FakeEmbedder:
    """Maps query strings to fixed vectors so tests skip MiniLM."""

    dimension = VECTOR_SIZE

    def __init__(self, query_vectors: dict[str, list[float]] | None = None) -> None:
        self.query_vectors = query_vectors or {}

    def embed_query(self, text: str) -> list[float]:
        if text in self.query_vectors:
            return self.query_vectors[text]
        return [0.0] * self.dimension


def _vector(*values: float) -> list[float]:
    filled = list(values)
    if len(filled) < VECTOR_SIZE:
        filled.extend([0.0] * (VECTOR_SIZE - len(filled)))
    return filled[:VECTOR_SIZE]


def _chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    filename: str = "paper.pdf",
    page_number: int = 1,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"filename": filename, "page_number": page_number},
        "embedding": embedding,
    }


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantStore(in_memory=True, vector_size=VECTOR_SIZE)
        transformer = _vector(1.0, 0.0, 0.0)
        baking = _vector(0.0, 1.0, 0.0)
        sports = _vector(0.0, 0.0, 1.0)
        self.store.upsert_chunks(
            [
                _chunk(
                    "paper_page_1_chunk_1",
                    "The Transformer architecture uses self-attention.",
                    transformer,
                    page_number=1,
                ),
                _chunk(
                    "paper_page_2_chunk_1",
                    "Chocolate cake is baked at 180 degrees.",
                    baking,
                    filename="recipes.pdf",
                    page_number=2,
                ),
                _chunk(
                    "paper_page_3_chunk_1",
                    "The team won the championship game.",
                    sports,
                    filename="sports.pdf",
                    page_number=3,
                ),
            ]
        )
        self.embedder = FakeEmbedder(
            {
                "How does the Transformer use attention?": transformer,
                "How do I bake a chocolate cake?": baking,
            }
        )
        self.retriever = Retriever(embedder=self.embedder, store=self.store)

    def test_empty_query_returns_empty_list(self) -> None:
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   \n"), [])

    def test_empty_collection_returns_empty_list(self) -> None:
        empty_store = QdrantStore(in_memory=True, vector_size=VECTOR_SIZE)
        retriever = Retriever(embedder=self.embedder, store=empty_store)
        self.assertEqual(retriever.retrieve("How does the Transformer use attention?"), [])

    def test_results_contain_text_score_and_metadata(self) -> None:
        results = self.retriever.retrieve(
            "How does the Transformer use attention?",
            top_k=1,
        )
        self.assertEqual(len(results), 1)
        hit = results[0]
        self.assertEqual(hit["text"], "The Transformer architecture uses self-attention.")
        self.assertEqual(hit["filename"], "paper.pdf")
        self.assertEqual(hit["page_number"], 1)
        self.assertEqual(hit["chunk_id"], "paper_page_1_chunk_1")
        self.assertIsInstance(hit["score"], float)
        self.assertGreater(hit["score"], 0.9)

    def test_results_are_ordered_most_similar_first(self) -> None:
        results = self.retriever.retrieve(
            "How does the Transformer use attention?",
            top_k=3,
        )
        self.assertEqual(len(results), 3)
        scores = [hit["score"] for hit in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0]["chunk_id"], "paper_page_1_chunk_1")

    def test_top_k_limits_the_number_of_hits(self) -> None:
        results = self.retriever.retrieve(
            "How does the Transformer use attention?",
            top_k=2,
        )
        self.assertEqual(len(results), 2)

    def test_score_threshold_filters_weak_matches(self) -> None:
        results = self.retriever.retrieve(
            "How does the Transformer use attention?",
            top_k=5,
            score_threshold=0.9,
        )
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(hit["score"] >= 0.9 for hit in results))
        self.assertEqual(results[0]["chunk_id"], "paper_page_1_chunk_1")

    def test_invalid_top_k_raises(self) -> None:
        with self.assertRaises(RetrievalError):
            self.retriever.retrieve("How does the Transformer use attention?", top_k=0)

    def test_qdrant_errors_are_wrapped(self) -> None:
        class BrokenStore:
            def search(self, *args, **kwargs):
                from app.vectorstore.qdrant_store import QdrantStoreError

                raise QdrantStoreError("client failed")

        retriever = Retriever(embedder=self.embedder, store=BrokenStore())  # type: ignore[arg-type]
        with self.assertRaises(RetrievalError):
            retriever.retrieve("How does the Transformer use attention?")


class RetrieverIntegrationTests(unittest.TestCase):
    """Uses the real MiniLM model with an in-memory Qdrant collection."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder = get_embedder()
        cls.store = QdrantStore(in_memory=True, vector_size=cls.embedder.dimension)
        chunks = [
            {
                "chunk_id": "paper_page_1_chunk_1",
                "text": "The Transformer architecture uses self-attention.",
                "metadata": {"filename": "paper.pdf", "page_number": 1},
            },
            {
                "chunk_id": "recipes_page_2_chunk_1",
                "text": "Chocolate cake is baked at 180 degrees in a metal pan.",
                "metadata": {"filename": "recipes.pdf", "page_number": 2},
            },
        ]
        cls.store.upsert_chunks(cls.embedder.embed_chunks(chunks))
        cls.retriever = Retriever(embedder=cls.embedder, store=cls.store)

    def test_query_uses_same_model_and_ranks_the_matching_chunk_first(self) -> None:
        results = self.retriever.retrieve(
            "How does the Transformer use attention?",
            top_k=2,
        )
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "paper_page_1_chunk_1")
        self.assertEqual(results[0]["filename"], "paper.pdf")
        self.assertEqual(results[0]["page_number"], 1)
        self.assertIn("self-attention", results[0]["text"])
        if len(results) > 1:
            self.assertGreaterEqual(results[0]["score"], results[1]["score"])


if __name__ == "__main__":
    unittest.main()
