"""Independent tests for the embedding layer (no FastAPI, Qdrant, or LLM)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.embedder import Embedder, get_embedder, get_embedding_dimension


def _chunk(
    text: str,
    chunk_id: str = "paper_page_1_chunk_1",
    filename: str = "paper.pdf",
    page_number: int = 1,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "filename": filename,
            "page_number": page_number,
        },
    }


class EmbedChunksTests(unittest.TestCase):
    embedder: Embedder

    @classmethod
    def setUpClass(cls) -> None:
        # Load the model once for the whole class — never inside a test loop.
        cls.embedder = get_embedder()

    def test_model_loads_successfully(self) -> None:
        self.assertIsNotNone(self.embedder.model)
        self.assertGreater(self.embedder.dimension, 0)
        self.assertEqual(get_embedding_dimension(), self.embedder.dimension)

    def test_normal_chunk_produces_numeric_embedding(self) -> None:
        results = self.embedder.embed_chunks(
            [_chunk("The Transformer architecture uses self-attention.")]
        )

        self.assertEqual(len(results), 1)
        embedding = results[0]["embedding"]
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertTrue(all(isinstance(value, float) for value in embedding))

    def test_embeddings_match_model_dimension(self) -> None:
        results = self.embedder.embed_chunks(
            [
                _chunk("Retrieval uses dense vectors.", chunk_id="a_page_1_chunk_1"),
                _chunk("Chunking preserves page metadata.", chunk_id="a_page_1_chunk_2"),
            ]
        )

        expected_dim = self.embedder.dimension
        self.assertGreater(expected_dim, 0)
        for item in results:
            self.assertEqual(len(item["embedding"]), expected_dim)

        lengths = {len(item["embedding"]) for item in results}
        self.assertEqual(lengths, {expected_dim})

    def test_multiple_chunks_are_embedded_in_a_batch(self) -> None:
        chunks = [
            _chunk("First passage about transformers.", chunk_id="paper_page_1_chunk_1"),
            _chunk("Second passage about retrieval.", chunk_id="paper_page_1_chunk_2"),
            _chunk("Third passage about embeddings.", chunk_id="paper_page_2_chunk_1"),
        ]

        results = self.embedder.embed_chunks(chunks)

        self.assertEqual(len(results), 3)
        self.assertEqual(
            [item["chunk_id"] for item in results],
            [chunk["chunk_id"] for chunk in chunks],
        )
        # Distinct texts should not all collapse to the identical vector.
        self.assertGreater(
            len({tuple(item["embedding"]) for item in results}),
            1,
        )

    def test_chunk_id_text_and_metadata_are_preserved(self) -> None:
        original = _chunk(
            "The Transformer architecture...",
            chunk_id="paper_page_1_chunk_1",
            filename="paper.pdf",
            page_number=1,
        )
        original_metadata = dict(original["metadata"])

        results = self.embedder.embed_chunks([original])

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item["chunk_id"], "paper_page_1_chunk_1")
        self.assertEqual(item["text"], "The Transformer architecture...")
        self.assertEqual(item["metadata"], original_metadata)
        self.assertIn("embedding", item)
        # Input is not mutated.
        self.assertNotIn("embedding", original)
        self.assertEqual(original["metadata"], original_metadata)

    def test_empty_input_is_handled_safely(self) -> None:
        self.assertEqual(self.embedder.embed_chunks([]), [])
        self.assertEqual(self.embedder.embed_chunks([_chunk("")]), [])
        self.assertEqual(self.embedder.embed_chunks([_chunk("   \n\t  ")]), [])

        mixed = [
            _chunk(""),
            _chunk("Only this chunk has real text.", chunk_id="paper_page_3_chunk_1"),
            _chunk("   "),
        ]
        results = self.embedder.embed_chunks(mixed)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "paper_page_3_chunk_1")
        self.assertEqual(results[0]["text"], "Only this chunk has real text.")


if __name__ == "__main__":
    unittest.main()
