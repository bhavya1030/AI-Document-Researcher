# Agentic RAG Research Assistant

Step-by-step Agentic RAG system. **This repository currently implements Step 1 (document ingestion and PDF parsing) and Step 2 (document chunking).**

Later steps (embeddings, Qdrant, hybrid retrieval, reranking, LangGraph, LLMs) are intentionally not included yet.

```
PDF upload
    ↓
Save to data/documents/
    ↓
Page-by-page PDF parsing (pypdf)
    ↓
Chunking (RecursiveCharacterTextSplitter)
    ↓
Embeddings (future)
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
│   │       └── chunker.py       # page text → retrieval chunks
│   ├── tests/
│   │   ├── test_parser.py       # parser unit tests
│   │   └── test_chunker.py      # chunker unit tests
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

Chunking is **not** connected to embeddings or Qdrant yet.

## Run tests

From `backend`:

```powershell
python -m unittest tests.test_parser
python -m unittest tests.test_chunker
```

Run both:

```powershell
python -m unittest discover -s tests -v
```

## What is intentionally not in this step

- Embeddings
- Qdrant / any vector database
- Full LangChain / LangGraph
- LLMs
- Frontend
- Wiring chunking into the upload API
