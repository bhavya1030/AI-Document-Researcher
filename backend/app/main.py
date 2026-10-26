"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.documents import router as documents_router

app = FastAPI(
    title="Agentic RAG Research Assistant",
    description="PDF upload, parse, chunk, embed, and store in Qdrant.",
    version="0.1.0",
)

app.include_router(documents_router)
