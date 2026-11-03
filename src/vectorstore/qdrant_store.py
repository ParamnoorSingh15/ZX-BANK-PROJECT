"""Qdrant vector store integration for ZX Bank RAG.

Responsibilities:
  * Create (or verify) the Qdrant collection with the correct vector size.
  * Store embedding vectors, text chunks and metadata in Qdrant.
  * Detect duplicates via content hashing so re-runs are idempotent.
  * Provide the LangChain ``QdrantVectorStore`` used by the retriever later.

Why Qdrant:
    Qdrant is a production-grade vector database that supports dense vectors,
    metadata filtering, MMR, hybrid search and payload storage. LangChain's
    ``QdrantVectorStore`` wraps it so we can store ``Document`` objects
    (text + metadata) directly.

Information flow:
    [chunk Documents]  -->  embed each chunk  -->  upsert into Qdrant
        (vector, payload = {text, metadata...})
    Qdrant collection  <--  ensure_collection()  (vector size from probe)
"""

from __future__ import annotations

import hashlib
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ..config.settings import get_settings
from ..embeddings.client import get_embeddings, probe_dimension


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hash of a text chunk for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _point_id(content_hash_value: str) -> str:
    """Derive a deterministic UUID v5 from a content hash.

    Using deterministic IDs makes re-indexing idempotent: upserting the same
    chunk twice overwrites the same point instead of creating duplicates.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, content_hash_value))


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient configured from environment settings."""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(
    client: Optional[QdrantClient] = None,
    embeddings=None,
    collection_name: Optional[str] = None,
) -> str:
    """Create the Qdrant collection if it does not exist.

    The vector size is discovered by probing the live embedding model so it
    matches what we will actually index and query with.

    Returns the collection name.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    if client is None:
        client = get_qdrant_client()
    if embeddings is None:
        embeddings = get_embeddings()

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return name

    dimension = probe_dimension(embeddings)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    return name


def _existing_hashes(
    client: QdrantClient, collection_name: str
) -> set:
    """Fetch all content_hash values already stored in Qdrant (paginated)."""
    hashes: set = set()
    offset = None
    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["content_hash"],
                with_vectors=False,
            )
        except Exception:
            # Collection may not exist yet.
            break
        for point in points:
            h = point.payload.get("content_hash") if point.payload else None
            if h:
                hashes.add(h)
        if next_offset is None:
            break
        offset = next_offset
    return hashes


def add_chunks(
    chunks: List[Document],
    client: Optional[QdrantClient] = None,
    embeddings=None,
    collection_name: Optional[str] = None,
    skip_duplicates: bool = True,
) -> int:
    """Add chunk Documents to Qdrant.

    Each chunk is embedded and upserted with its text and metadata in the
    Qdrant payload. Returns the number of chunks actually added (skipping
    duplicates when enabled).

    Duplicate detection is done efficiently: existing content hashes are
    fetched once (paginated), then filtered client-side. New chunks are
    embedded in a single batch and upserted with deterministic IDs derived
    from their content hash, so re-running ingestion is idempotent.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    if client is None:
        client = get_qdrant_client()
    if embeddings is None:
        embeddings = get_embeddings()

    ensure_collection(client, embeddings, name)

    # Annotate each chunk with its content hash.
    for chunk in chunks:
        chunk.metadata["content_hash"] = content_hash(chunk.page_content)

    # Skip empty chunks (no usable text).
    filtered = [
        chunk for chunk in chunks if chunk.page_content.strip()
    ]

    if skip_duplicates:
        existing = _existing_hashes(client, name)
        filtered = [
            chunk for chunk in filtered
            if chunk.metadata["content_hash"] not in existing
        ]
    else:
        filtered = list(chunks)

    if not filtered:
        return 0

    # Deduplicate within this batch (keep first occurrence of each hash).
    seen: set = set()
    unique_filtered = []
    for chunk in filtered:
        h = chunk.metadata["content_hash"]
        if h in seen:
            continue
        seen.add(h)
        unique_filtered.append(chunk)

    # Batch-embed chunks (smaller batches so Ollama stays responsive) and
    # upsert incrementally. Deterministic IDs make re-runs resume safely.
    EMBED_BATCH = 64
    UPSERT_BATCH = 100
    added = 0
    points: List[PointStruct] = []

    for start in range(0, len(unique_filtered), EMBED_BATCH):
        batch_chunks = unique_filtered[start : start + EMBED_BATCH]
        batch_texts = [chunk.page_content for chunk in batch_chunks]
        batch_vectors = embeddings.embed_documents(batch_texts)
        for chunk, vector in zip(batch_chunks, batch_vectors):
            points.append(
                PointStruct(
                    id=_point_id(chunk.metadata["content_hash"]),
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        **chunk.metadata,
                    },
                )
            )
            if len(points) >= UPSERT_BATCH:
                client.upsert(collection_name=name, points=points)
                added += len(points)
                points = []
                print(f"  ... indexed {added} chunks", flush=True)

    if points:
        client.upsert(collection_name=name, points=points)
        added += len(points)
        print(f"  ... indexed {added} chunks total", flush=True)

    return added


def get_vector_store(
    client: Optional[QdrantClient] = None,
    embeddings=None,
    collection_name: Optional[str] = None,
) -> QdrantVectorStore:
    """Return a ``QdrantVectorStore`` bound to the configured collection.

    Used by the retrieval layer (Branch 3) for similarity search and MMR.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    if client is None:
        client = get_qdrant_client()
    if embeddings is None:
        embeddings = get_embeddings()
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=embeddings,
    )
