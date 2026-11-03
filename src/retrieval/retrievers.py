"""Retrievers for ZX Bank RAG.

Why a retriever:
    A LangChain ``Retriever`` takes a query string and returns relevant
    ``Document`` objects. It abstracts the vector-store search so the RAG
    chain can swap dense search, MMR, or filtered search without changing
    the generation logic.

Retrievers implemented:
    * DenseRetriever      - cosine similarity search in Qdrant
    * MMRRetriever        - Maximal Marginal Relevance (diversifies results)
    * MetadataFilteredRetriever - similarity search with a metadata filter

Information flow:
    user query  -->  retriever.invoke(query)
        -->  QdrantClient.query_points (dense vector search)
        -->  [Document(page_content, metadata)]
    Retrieved docs  -->  context construction  -->  ChatPromptTemplate

Why direct QdrantClient.query_points instead of QdrantVectorStore.similarity_search:
    Qdrant only builds its HNSW index once a collection crosses the
    indexing_threshold (default 10,000 points). With fewer points, raw
    similarity_search can return empty hits. Using ``QdrantClient.query_points``
    directly performs brute-force cosine search over all stored vectors, so
    retrieval works regardless of collection size. The downside is linear
    scan cost, which is fine for this dataset scale.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import Field
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from src.vectorstore.qdrant_store import (
    get_qdrant_client,
)
from src.embeddings.client import get_embeddings
from src.config.settings import get_settings


def _document_from_point(point) -> Document:
    """Convert a Qdrant point into a LangChain Document."""
    payload = point.payload or {}
    text = payload.pop("text", "")
    return Document(page_content=text, metadata=dict(payload))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class DenseRetriever(BaseRetriever):
    """Dense cosine similarity search over Qdrant (brute-force via query_points)."""

    k: int = Field(default=4)
    collection_name: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        settings = get_settings()
        name = self.collection_name or settings.qdrant_collection
        client = get_qdrant_client()
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)
        result = client.query_points(
            collection_name=name,
            query=query_vector,
            limit=self.k,
            with_payload=True,
            with_vectors=False,
        )
        return [_document_from_point(h) for h in result.points]


class MMRRetriever(BaseRetriever):
    """Maximal Marginal Relevance retriever.

    MMR balances relevance to the query with diversity among results. We fetch
    ``fetch_k`` candidates via dense search, then re-rank with MMR using
    cosine similarity between candidates and the query, penalizing candidates
    similar to already-selected ones.
    """

    k: int = Field(default=4)
    fetch_k: int = Field(default=20)
    lambda_mult: float = Field(default=0.5)
    collection_name: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        settings = get_settings()
        name = self.collection_name or settings.qdrant_collection
        client = get_qdrant_client()
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)

        result = client.query_points(
            collection_name=name,
            query=query_vector,
            limit=self.fetch_k,
            with_payload=True,
            with_vectors=True,
        )
        hits = result.points
        if not hits:
            return []

        query_vec = np.array(query_vector, dtype=np.float32)
        # Candidate vectors and similarities to query.
        candidates = []
        for h in hits:
            vec = h.vector
            if vec is None:
                continue
            cand_vec = np.array(vec, dtype=np.float32)
            sim_q = _cosine_similarity(query_vec, cand_vec)
            candidates.append((h, sim_q, cand_vec))

        # MMR selection.
        selected: List = []
        remaining = list(candidates)
        while remaining and len(selected) < self.k:
            if not selected:
                best = max(remaining, key=lambda c: c[1])
            else:
                best = None
                best_score = -float("inf")
                for h, sim_q, cand_vec in remaining:
                    max_sim_to_selected = 0.0
                    for sh, _, _ in selected:
                        sv = sh.vector
                        if sv is None:
                            continue
                        sv_arr = np.array(sv, dtype=np.float32)
                        sim_s = _cosine_similarity(cand_vec, sv_arr)
                        max_sim_to_selected = max(max_sim_to_selected, sim_s)
                    mmr = (
                        self.lambda_mult * sim_q
                        - (1 - self.lambda_mult) * max_sim_to_selected
                    )
                    if mmr > best_score:
                        best_score = mmr
                        best = (h, sim_q, cand_vec)
                if best is None:
                    break
            selected.append(best[0])
            remaining = [c for c in remaining if c[0].id != best[0].id]

        return [_document_from_point(h) for h, _, _ in selected]


class MetadataFilteredRetriever(BaseRetriever):
    """Similarity search with a Qdrant metadata filter.

    Useful for scoping retrieval to a specific document, format, branch,
    etc. The filter is a Qdrant filter dict passed at retrieval time.
    """

    k: int = Field(default=4)
    filter: Optional[Dict[str, Any]] = Field(default=None)
    collection_name: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        settings = get_settings()
        name = self.collection_name or settings.qdrant_collection
        client = get_qdrant_client()
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(query)

        qdrant_filter = None
        if self.filter:
            conditions = []
            for key, value in self.filter.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        result = client.query_points(
            collection_name=name,
            query=query_vector,
            limit=self.k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )
        return [_document_from_point(h) for h in result.points]


def get_retriever(
    strategy: str = "dense",
    k: int = 4,
    collection_name: Optional[str] = None,
    **kwargs,
) -> BaseRetriever:
    """Factory to build a retriever by strategy name.

    Args:
        strategy: One of "dense", "mmr", "filtered".
        k: Number of documents to return.
        collection_name: Qdrant collection name.
        **kwargs: Strategy-specific options (e.g. filter for "filtered",
            fetch_k/lambda_mult for "mmr").

    Returns:
        A LangChain ``BaseRetriever`` instance.
    """
    if strategy == "mmr":
        return MMRRetriever(
            k=k,
            fetch_k=kwargs.get("fetch_k", 20),
            lambda_mult=kwargs.get("lambda_mult", 0.5),
            collection_name=collection_name,
        )
    if strategy == "filtered":
        return MetadataFilteredRetriever(
            k=k,
            filter=kwargs.get("filter"),
            collection_name=collection_name,
        )
    # Default: dense similarity search.
    return DenseRetriever(
        k=k, collection_name=collection_name
    )
