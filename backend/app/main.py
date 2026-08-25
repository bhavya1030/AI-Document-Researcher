"""FastAPI application entrypoint for Step 1: PDF ingestion."""

from fastapi import FastAPI

from app.api.documents import router as documents_router

app = FastAPI(
    title="Agentic RAG Research Assistant",
    description="Step 1: Document ingestion and PDF parsing.",
    version="0.1.0",
)

app.include_router(documents_router)
