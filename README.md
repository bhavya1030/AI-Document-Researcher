# Agentic RAG Research Assistant

Step-by-step Agentic RAG system. **This repository currently implements Step 1 (document ingestion and PDF parsing), Step 2 (document chunking), and Step 3 (embedding generation).**

Later steps (Qdrant, retrieval, reranking, LangGraph, LLMs) are intentionally not included yet.

```
PDF upload
    ↓
Save to data/documents/
    ↓
Page-by-page PDF parsing (pypdf)
    ↓
Chunking (RecursiveCharacterTextSplitter)
    ↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    ↓
Qdrant (future)
```

## Project structure

```
agentic-rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── documents.py     # POST /documents/upload
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       ├── parser.py        # PDF text extraction
│   │       ├── chunker.py       # page text → retrieval chunks
│   │       └── embedder.py      # chunks → dense vectors
│   ├── tests/
│   │   ├── test_parser.py       # parser unit tests
│   │   ├── test_chunker.py      # chunker unit tests
│   │   └── test_embedder.py     # embedding unit tests
│   └── requirements.txt
├── data/
│   └── documents/               # saved uploads
└── README.md
```

## Setup

Use Python 3.10 or newer.

### 1. Create a virtual environment

From the `agentic-rag` directory:

**Windows (PowerShell)**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Dependencies for this step:

- `fastapi` — API framework
- `uvicorn` — ASGI server
- `pypdf` — PDF text extraction
- `python-multipart` — required for file uploads
- `langchain-text-splitters` — text chunking (`RecursiveCharacterTextSplitter` only; not the full LangChain framework)
- `sentence-transformers` — local embedding model (`all-MiniLM-L6-v2`)

## Start the API

From the `backend` directory, with the virtual environment activated:

```powershell
uvicorn app.main:app --reload
```

The server listens on [http://127.0.0.1:8000](http://127.0.0.1:8000).

Interactive docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API endpoint

`POST /documents/upload`

Accepts a single PDF file (`multipart/form-data`, field name: `file`).

- Validates that the upload is a PDF
- Saves the file to `data/documents/`
- Parses text **page by page** with `pypdf`
- Returns JSON with filename, page count, and per-page text

Chunking is a separate ingestion module (`chunker.py`). It is not wired into the upload route yet.

### Expected success response

```json
{
  "filename": "example.pdf",
  "total_pages": 3,
  "pages": [
    {
      "page_number": 1,
      "text": "Text extracted from page 1..."
    },
    {
      "page_number": 2,
      "text": "Text extracted from page 2..."
    },
    {
      "page_number": 3,
      "text": ""
    }
  ]
}
```

Pages with no extractable text still appear in `pages`, with `"text": ""`.

### Error cases

| Situation | HTTP status | Detail |
| --- | --- | --- |
| Missing / non-`.pdf` filename | 400 | Only PDF files are accepted |
| Empty upload | 400 | Uploaded file is empty |
| File does not start with PDF magic bytes | 400 | File is not a valid PDF |
| Corrupted or unreadable PDF | 400 | Unable to parse PDF ... |
| Encrypted PDF | 400 | Encrypted PDFs are not supported |

## Test the upload endpoint with Swagger UI

1. Start the server (`uvicorn app.main:app --reload` from `backend`).
2. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
3. Expand `POST /documents/upload`.
4. Click **Try it out**.
5. Click **Choose File** and select a `.pdf`.
6. Click **Execute**.
7. Confirm:
   - HTTP 200
   - `filename` matches the uploaded file
   - `total_pages` is correct
   - `pages` is a list of `{ "page_number", "text" }` objects
8. Confirm the same PDF now exists under `agentic-rag/data/documents/`.

You can also call it with curl:

```powershell
curl -X POST "http://127.0.0.1:8000/documents/upload" -F "file=@C:\path\to\example.pdf"
```

## Step 2: Document chunking

`backend/app/ingestion/chunker.py` takes the page-level output from `parser.py` and splits each page into smaller text chunks.

Defaults (passed as function arguments, not scattered through the code):

- `chunk_size = 1000`
- `chunk_overlap = 150`

Behavior:

- Empty or whitespace-only pages are skipped
- Extra whitespace is cleaned before splitting
- Each chunk keeps `filename` and `page_number`
- Chunk IDs are deterministic, for example `example_page_4_chunk_2`

Example chunk:

```json
{
  "chunk_id": "example_page_4_chunk_2",
  "text": "...",
  "metadata": {
    "filename": "example.pdf",
    "page_number": 4
  }
}
```

Chunking is **not** connected to the upload API or to Qdrant yet.

## Step 3: Embedding generation

`backend/app/ingestion/embedder.py` takes the structured chunks from `chunker.py` and attaches a dense vector to each one.

### What embeddings are

An embedding is a list of floating-point numbers that represents the *meaning* of a piece of text. Similar sentences land close together in that vector space; unrelated sentences land far apart.

Example:

```
text:  "The Transformer architecture uses self-attention."
   ↓
embedding: [0.021, -0.084, 0.113, ...]   # 384 numbers
```

### Why RAG needs embeddings

Retrieval does not search raw strings. It embeds the user question with the same model, then finds document chunks whose vectors are nearest to the question vector. Those chunks become the context for the LLM in a later step.

### Why `all-MiniLM-L6-v2`

- Small enough for a 16 GB laptop (CPU is fine)
- No API key and no network call at inference time after the first model download
- Well-tested for semantic search
- Output width is **384**, read at runtime from the loaded model via `get_embedding_dimension()` / `Embedder.dimension` — not hardcoded in the rest of the app

### Why the model runs locally

Embeddings are generated on every ingested chunk. A local SentenceTransformer avoids per-chunk API cost, latency, and leaking document text to a third-party embedding API.

### Why the model is loaded once

Loading MiniLM from disk (and the first Hugging Face download) is the expensive part. `get_embedder()` keeps a single process-wide `Embedder`. Do not construct `SentenceTransformer` inside a per-chunk or per-request loop.

### Why batching

`Embedder.embed_chunks` sends many chunk texts through `model.encode(..., batch_size=32)` in one call. That uses the model more efficiently than encoding each string separately.

### Normalization

Embeddings are **L2-normalized** (`normalize_embeddings=True`). For this model, later similarity search in Qdrant should use cosine similarity. On unit-length vectors, cosine similarity equals a dot product, so Qdrant can use cosine (or dot product on already-normalized vectors) without a second normalization pass.

### Output shape

```json
{
  "chunk_id": "paper_page_1_chunk_1",
  "text": "The Transformer architecture...",
  "metadata": {
    "filename": "paper.pdf",
    "page_number": 1
  },
  "embedding": [0.123, -0.234]
}
```

Empty or whitespace-only chunks are skipped. Original `chunk_id`, `text`, and `metadata` are preserved. Vectors stay in Python memory; they are not written to Qdrant yet.

### How Qdrant will use this (next step)

Each returned object is one Qdrant point: the `embedding` is the vector, and `chunk_id` / `text` / `metadata` become the payload. Collection vector size must equal `get_embedding_dimension()`.

### Pipeline

```
PDF
 → Parser     (page text)
 → Chunker    (smaller passages + metadata)
 → Embedder   (same passages + 384-d vectors)
```

The embedder is **not** wired into `POST /documents/upload` yet.

## Run tests

From `backend`, with the virtual environment activated:

```powershell
pip install -r requirements.txt
```

```powershell
python -m unittest tests.test_parser
python -m unittest tests.test_chunker
python -m unittest tests.test_embedder
```

The first `test_embedder` run downloads `all-MiniLM-L6-v2` (one-time). Later runs reuse the local cache and the process-wide model.

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

## What is intentionally not in this step

- Qdrant / any vector database
- Retrieval / similarity search
- Reranking
- Full LangChain / LangGraph
- LLMs
- Frontend
- Wiring chunking or embeddings into the upload API
