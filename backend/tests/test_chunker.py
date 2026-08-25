"""Independent tests for the document chunker (no FastAPI, embeddings, or Qdrant)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.chunker import chunk_pages


def _page(text: str, page_number: int = 1, filename: str = "example.pdf") -> dict:
    return {
        "page_number": page_number,
        "text": text,
        "metadata": {
            "filename": filename,
            "page_number": page_number,
            "extracted_text": text,
        },
    }


class ChunkPagesTests(unittest.TestCase):
    def test_splits_large_text_into_multiple_chunks(self) -> None:
        text = "Retrieval augmented generation uses smaller passages. " * 80
        chunks = chunk_pages([_page(text)], chunk_size=100, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk["text"].strip())
            self.assertLessEqual(len(chunk["text"]), 150)

    def test_chunk_overlap_is_applied(self) -> None:
        words = [f"word{i:03d}" for i in range(80)]
        text = " ".join(words)
        chunks = chunk_pages([_page(text)], chunk_size=80, chunk_overlap=25)

        self.assertGreaterEqual(len(chunks), 2)

        first_words = chunks[0]["text"].split()
        second_words = chunks[1]["text"].split()
        overlap = [word for word in first_words if word in second_words]

        self.assertGreaterEqual(
            len(overlap),
            1,
            "Consecutive chunks should share overlapping tokens",
        )
        self.assertEqual(second_words[: len(overlap)], overlap[-len(overlap) :])

    def test_page_metadata_is_preserved(self) -> None:
        text = "Page four has enough text to split into several retrieval chunks. " * 40
        chunks = chunk_pages(
            [_page(text, page_number=4, filename="paper.pdf")],
            chunk_size=120,
            chunk_overlap=20,
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["metadata"]["page_number"], 4)

    def test_filename_metadata_is_preserved(self) -> None:
        chunks = chunk_pages(
            [_page("A short page of notes.", filename="research-notes.pdf")],
            chunk_size=1000,
            chunk_overlap=150,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["filename"], "research-notes.pdf")

    def test_chunk_ids_are_unique_and_deterministic(self) -> None:
        pages = [
            _page("Alpha content about transformers. " * 30, page_number=1, filename="example.pdf"),
            _page("Beta content about retrieval. " * 30, page_number=2, filename="example.pdf"),
        ]

        first = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
        second = chunk_pages(pages, chunk_size=100, chunk_overlap=20)

        ids = [chunk["chunk_id"] for chunk in first]
        self.assertEqual(ids, [chunk["chunk_id"] for chunk in second])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(ids[0].startswith("example_page_1_chunk_"))
        self.assertIn("example_page_2_chunk_1", ids)
        self.assertEqual(first[0]["chunk_id"], "example_page_1_chunk_1")

    def test_empty_pages_are_ignored(self) -> None:
        pages = [
            _page("", page_number=1),
            _page("   \n\n  ", page_number=2),
            _page("Only this page has real text.", page_number=3),
        ]
        chunks = chunk_pages(pages)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["page_number"], 3)
        self.assertEqual(chunks[0]["chunk_id"], "example_page_3_chunk_1")
        self.assertIn("Only this page has real text.", chunks[0]["text"])

    def test_whitespace_is_cleaned_before_chunking(self) -> None:
        messy = "  Hello   world.  \n\n\n\n  Next   sentence.  "
        chunks = chunk_pages([_page(messy)])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Hello world.\n\nNext sentence.")


if __name__ == "__main__":
    unittest.main()
