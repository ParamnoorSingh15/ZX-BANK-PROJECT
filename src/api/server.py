"""FastAPI backend for ZX Bank RAG.

Endpoints:
    POST   /ask              - Ask a question, get grounded answer + sources
    POST   /ask/stream       - (placeholder) streaming answers
    GET    /sources/{doc_id} - Get document source details
    POST   /documents/upload - Upload a document for ingestion
    GET    /health           - Health check
    GET    /collections      - List Qdrant collections and stats

Why FastAPI:
    Async web framework with automatic OpenAPI/Swagger docs, Pydantic-based
    request/response validation, and easy dependency injection. Ideal for
    exposing a RAG pipeline as a REST API.

Information flow:
    HTTP request  -->  FastAPI endpoint  -->  RAG chain (retrieve + generate)
        -->  JSON response {answer, sources, retrieved_documents}
"""

from __future__ import annotations

import os
import tempfile
import shutil
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config.settings import get_settings
from src.chains import BasicRAGChain, ConversationalRAGChain, extract_sources
from src.retrieval.retrievers import get_retriever, reset_corpus_cache
from src.vectorstore.qdrant_store import (
    get_qdrant_client,
    get_vector_store,
    add_chunks,
    ensure_collection,
    content_hash,
)
from src.ingestion.pipeline import load_and_split_dataset
from src.ingestion.loaders import load_document
from src.ingestion.cleaners import clean_documents
from src.ingestion.splitter import split_documents
from src.embeddings.client import get_embeddings

settings = get_settings()

# In-memory chat history store (session_id -> list of (role, content)).
# For production, replace with Redis or a database.
CHAT_HISTORY: Dict[str, List[tuple]] = {}

app = FastAPI(
    title="ZX Bank RAG API",
    description="Grounded banking Q&A over ZX Bank documents using LangChain, Ollama and Qdrant.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared chain instances (created lazily on first request).
_basic_chain: Optional[BasicRAGChain] = None
_conv_chain: Optional[ConversationalRAGChain] = None


def _get_basic_chain() -> BasicRAGChain:
    global _basic_chain
    if _basic_chain is None:
        _basic_chain = BasicRAGChain(
            retriever_strategy=settings.retrieval_strategy,
            k=settings.retrieval_k,
            collection_name=settings.qdrant_collection,
        )
    return _basic_chain


def _get_conv_chain() -> ConversationalRAGChain:
    global _conv_chain
    if _conv_chain is None:
        _conv_chain = ConversationalRAGChain(
            retriever_strategy=settings.retrieval_strategy,
            k=settings.retrieval_k,
            collection_name=settings.qdrant_collection,
        )
    return _conv_chain


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    strategy: str = "hybrid"
    k: int = 8
    use_conversation: bool = False
    session_id: Optional[str] = None


class SourceRef(BaseModel):
    title: str
    source: str
    format: str
    page: Optional[int] = None
    slide: Optional[int] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    content_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceRef]
    strategy: str
    k: int
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    collection: str
    collection_exists: bool
    vector_count: int


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    collection: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=Dict[str, str])
def root():
    return {
        "name": "ZX Bank RAG API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        exists = settings.qdrant_collection in collections
        vector_count = 0
        if exists:
            info = client.get_collection(settings.qdrant_collection)
            vector_count = info.points_count or 0
        return HealthResponse(
            status="ok",
            collection=settings.qdrant_collection,
            collection_exists=exists,
            vector_count=vector_count,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {exc}")


@app.get("/collections")
def list_collections():
    client = get_qdrant_client()
    cols = client.get_collections().collections
    result = []
    for c in cols:
        try:
            info = client.get_collection(c.name)
            result.append(
                {
                    "name": c.name,
                    "points_count": info.points_count,
                    "vector_size": info.config.params.vectors.size
                    if info.config.params.vectors
                    else None,
                    "distance": str(info.config.params.vectors.distance)
                    if info.config.params.vectors
                    else None,
                }
            )
        except Exception:
            result.append({"name": c.name})
    return {"collections": result}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question and get a detailed, grounded answer with citations."""
    try:
        session_id = req.session_id
        if req.use_conversation and not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        if req.use_conversation and session_id:
            if (
                req.strategy != settings.retrieval_strategy
                or req.k != settings.retrieval_k
            ):
                chain = ConversationalRAGChain(
                    retriever_strategy=req.strategy,
                    k=req.k,
                    collection_name=settings.qdrant_collection,
                )
            else:
                chain = _get_conv_chain()
            history = CHAT_HISTORY.get(session_id, [])
            answer, docs = chain.invoke(req.question, chat_history=history)
            # Keep only the recent turns so the prompt does not grow forever.
            CHAT_HISTORY[session_id] = (history + [
                ("user", req.question),
                ("assistant", answer),
            ])[-12:]
        else:
            if (
                req.strategy != settings.retrieval_strategy
                or req.k != settings.retrieval_k
            ):
                chain = BasicRAGChain(
                    retriever_strategy=req.strategy,
                    k=req.k,
                    collection_name=settings.qdrant_collection,
                )
            else:
                chain = _get_basic_chain()
            answer, docs = chain.invoke(req.question)

        sources = [SourceRef(**source) for source in extract_sources(docs)]

        return AskResponse(
            answer=answer,
            sources=sources,
            strategy=req.strategy,
            k=req.k,
            session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/documents/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    """Upload a document, ingest it, and index its chunks into Qdrant.

    Supports .pdf, .html, .docx, .pptx, .md files.
    """
    filename = file.filename or "uploaded"
    ext = os.path.splitext(filename)[1].lower()
    allowed = {".pdf", ".html", ".htm", ".docx", ".pptx", ".md"}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}",
        )

    # Save upload to a temp file.
    suffix = ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Load, clean, split.
        docs = load_document(tmp_path)
        docs = clean_documents(docs)
        chunks = split_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        # Index into Qdrant.
        embeddings = get_embeddings()
        added = add_chunks(
            chunks,
            collection_name=settings.qdrant_collection,
            embeddings=embeddings,
        )
        # The hybrid retriever caches payloads in memory; refresh it so the new
        # document is immediately searchable.
        reset_corpus_cache(settings.qdrant_collection)
        return UploadResponse(
            filename=filename,
            chunks_indexed=added,
            collection=settings.qdrant_collection,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.delete("/conversations/{session_id}")
def clear_conversation(session_id: str):
    CHAT_HISTORY.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
