"""Small process settings. Paths and names live here, not in every module."""

from __future__ import annotations

import os
from pathlib import Path

# agentic-rag/  (this file is backend/app/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

QDRANT_PATH = Path(os.environ.get("QDRANT_PATH", str(PROJECT_ROOT / "data" / "qdrant")))
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "document_chunks")
