"""End-to-end ingestion: parse → chunk → embed → Qdrant (in-memory, no MiniLM)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient

from app.ingestion.chunker import chunk_pages
from app.ingestion.parser import parse_pdf
from app.ingestion.pipeline import ingest_pages
from app.main import app
from app.vectorstore.qdrant_store import QdrantStore
from tests.test_parser import build_simple_pdf

VECTOR_SIZE = 8


class FakeEmbedder:
    dimension = VECTOR_SIZE

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        embedded = []
        for index, chunk in enumerate(chunks):
            vector = [0.0] * VECTOR_SIZE
            vector[0] = float(index + 1)
            embedded.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": dict(chunk.get("metadata") or {}),
                    "embedding": vector,
                }
            )
        return embedded


def _write_temp_pdf(payload: bytes, suffix: str = ".pdf") -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(payload)
    handle.close()
    return Path(handle.name)


class IngestPagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder()
        self.store = QdrantStore(in_memory=True, vector_size=VECTOR_SIZE)

    def test_parse_chunk_embed_and_upsert_preserves_metadata(self) -> None:
        path = _write_temp_pdf(
            build_simple_pdf(["Transformer attention", "Retrieval uses vectors"]),
            suffix="_paper.pdf",
        )
        self.addCleanup(path.unlink)

        pages = parse_pdf(path)
        result = ingest_pages(pages, embedder=self.embedder, store=self.store)

        self.assertEqual(len(pages), 2)
        self.assertGreaterEqual(result["chunk_count"], 2)
        self.assertEqual(result["vector_count"], result["chunk_count"])
        self.assertEqual(self.store.count(), result["vector_count"])
        self.assertTrue(self.store.collection_exists())

        first_id = chunk_pages(pages)[0]["chunk_id"]
        stored = self.store.get_by_chunk_id(first_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["filename"], path.name)
        self.assertEqual(stored["page_number"], 1)
        self.assertEqual(stored["chunk_id"], first_id)
        self.assertIn("Transformer", stored["text"])

    def test_empty_pages_store_nothing(self) -> None:
        result = ingest_pages(
            [{"page_number": 1, "text": "   ", "metadata": {"filename": "empty.pdf"}}],
            embedder=self.embedder,
            store=self.store,
        )
        self.assertEqual(result["chunk_count"], 0)
        self.assertEqual(result["vector_count"], 0)
        self.assertEqual(self.store.count(), 0)


class UploadApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.embedder = FakeEmbedder()
        self.store = QdrantStore(in_memory=True, vector_size=VECTOR_SIZE)
        self.client = TestClient(app)

        self.dir_patch = patch(
            "app.api.documents.DOCUMENTS_DIR",
            Path(self.tmp.name),
        )
        self.embedder_patch = patch(
            "app.ingestion.pipeline.get_embedder",
            return_value=self.embedder,
        )
        self.store_patch = patch(
            "app.ingestion.pipeline.get_qdrant_store",
            return_value=self.store,
        )
        self.dir_patch.start()
        self.embedder_patch.start()
        self.store_patch.start()
        self.addCleanup(self.dir_patch.stop)
        self.addCleanup(self.embedder_patch.stop)
        self.addCleanup(self.store_patch.stop)

    def test_upload_runs_full_ingestion_and_returns_counts(self) -> None:
        payload = build_simple_pdf(["Hello World from page one"])
        response = self.client.post(
            "/documents/upload",
            files={"file": ("notes.pdf", payload, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["filename"], "notes.pdf")
        self.assertEqual(body["total_pages"], 1)
        self.assertGreaterEqual(body["chunk_count"], 1)
        self.assertEqual(body["vector_count"], body["chunk_count"])
        self.assertEqual(self.store.count(), body["vector_count"])

        stored = self.store.get_by_chunk_id("notes_page_1_chunk_1")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["filename"], "notes.pdf")
        self.assertEqual(stored["page_number"], 1)
        self.assertEqual(stored["chunk_id"], "notes_page_1_chunk_1")
        saved = Path(self.tmp.name) / "notes.pdf"
        self.assertTrue(saved.is_file())

    def test_upload_rejects_non_pdf(self) -> None:
        response = self.client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"not a pdf", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()
