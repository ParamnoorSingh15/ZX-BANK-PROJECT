"""Retrievers for ZX Bank RAG.

Why a retriever:
    A LangChain ``Retriever`` takes a query string and returns relevant
    ``Document`` objects. It abstracts the vector-store search so the RAG
    chain can swap strategies without changing the generation logic.

Retrievers implemented:
    * HybridRetriever     - dense cosine search + BM25 keyword search (default)
    * DenseRetriever      - cosine similarity search in Qdrant
    * MMRRetriever        - Maximal Marginal Relevance (diversifies results)
    * MetadataFilteredRetriever - similarity search with a metadata filter

Why hybrid retrieval:
    ZX Bank questions often contain exact terms ("3.50%", "IFSC", "Mumbai",
    "locker") that dense embeddings blur together. BM25 catches those literal
    matches while dense search catches paraphrases. Merging both ranks fixes
    cases where the vector search returned a section title but never the table
    of rates underneath it.

Why deduplicate at search time:
    The dataset stores each document in five formats (pdf, html, docx, pptx,
    md). All formats are indexed so no content is lost, but without dedupe the
    top-8 slots fill with five copies of the same paragraph. We remove
    near-duplicate chunks and cap how many chunks one document may contribute.

Why neighbour expansion:
    A single chunk can contain a heading and part of a table. Expanding a hit
    with the chunks immediately before/after it gives the LLM the full section
    ("whole context") instead of a fragment.

Why direct QdrantClient.query_points instead of QdrantVectorStore.similarity_search:
    Qdrant only builds its HNSW index once a collection crosses the
    indexing_threshold (default 10,000 points). With fewer points, raw
    similarity_search can return empty hits. Using ``QdrantClient.query_points``
    directly performs brute-force cosine search over all stored vectors, so
    retrieval works regardless of collection size.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import Field
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from src.vectorstore.qdrant_store import get_qdrant_client
from src.embeddings.client import get_embeddings
from src.config.settings import get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,%][0-9]+)*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "our", "the", "their", "this", "to", "was", "we",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
}


def _stem(token: str) -> str:
    """Very small suffix stemmer so 'headquarters' matches 'headquartered'."""
    for suffix in ("ations", "ation", "ings", "ing", "ers", "er", "ies", "ied", "es", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            base = token[: -len(suffix)]
            if suffix in {"ies", "ied"}:
                base += "y"
            return base
    return token


def _tokenize(text: str) -> List[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _STOPWORDS:
            continue
        tokens.append(_stem(raw))
    return tokens


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _document_from_point(point) -> Document:
    """Convert a Qdrant point into a LangChain Document."""
    payload = dict(point.payload or {})
    text = payload.pop("text", "")
    return Document(page_content=text, metadata=payload)


@dataclass
class _CorpusRecord:
    id: str
    text: str
    metadata: dict
    tokens: List[str] = field(default_factory=list)
    token_set: Set[str] = field(default_factory=set)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", "unknown"))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    def as_document(self) -> Document:
        payload = dict(self.metadata)
        return Document(page_content=self.text, metadata=payload)


class _BM25:
    """Lightweight BM25 index built from Qdrant payloads."""

    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_len) / max(1, len(tokenized_docs))

        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter = Counter()
        for index, tokens in enumerate(tokenized_docs):
            freq = Counter(tokens)
            for token, count in freq.items():
                self.postings[token].append((index, count))
            doc_freq.update(freq.keys())

        total = max(1, len(tokenized_docs))
        self.idf = {
            token: math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for token, df in doc_freq.items()
        }

    def scores(self, query_tokens: List[str]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, freq in self.postings.get(token, []):
                dl = self.doc_len[index]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[index] += idf * (freq * (self.k1 + 1.0)) / denom
        return scores


@dataclass
class _CorpusIndex:
    records: List[_CorpusRecord]
    bm25: _BM25
    by_id: Dict[str, _CorpusRecord]
    by_source: Dict[Tuple[str, str], List[_CorpusRecord]]
    position_by_id: Dict[str, int]


_CORPUS_CACHE: Dict[str, _CorpusIndex] = {}
_CORPUS_LOCK = threading.Lock()


def reset_corpus_cache(collection_name: Optional[str] = None) -> None:
    """Drop the in-memory corpus cache (call after re-indexing)."""
    with _CORPUS_LOCK:
        if collection_name is None:
            _CORPUS_CACHE.clear()
        else:
            _CORPUS_CACHE.pop(collection_name, None)


def _build_corpus(collection_name: str) -> _CorpusIndex:
    client = get_qdrant_client()
    records: List[_CorpusRecord] = []
    by_id: Dict[str, _CorpusRecord] = {}
    by_source: Dict[Tuple[str, str], List[_CorpusRecord]] = defaultdict(list)

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            if not text.strip():
                continue
            record = _CorpusRecord(
                id=str(point.id),
                text=text,
                metadata=payload,
                tokens=_tokenize(text),
            )
            record.token_set = set(record.tokens)
            record.metadata.setdefault("point_id", str(point.id))
            records.append(record)
            by_id[record.id] = record
            source_key = (
                str(payload.get("source", "")),
                str(payload.get("format", "")),
            )
            by_source[source_key].append(record)
        if offset is None:
            break

    for source_records in by_source.values():
        source_records.sort(
            key=lambda record: int(record.metadata.get("chunk_index") or 0)
        )

    bm25 = _BM25([record.tokens for record in records])
    position_by_id = {record.id: index for index, record in enumerate(records)}
    return _CorpusIndex(
        records=records,
        bm25=bm25,
        by_id=by_id,
        by_source=dict(by_source),
        position_by_id=position_by_id,
    )


def _get_corpus(collection_name: str) -> _CorpusIndex:
    with _CORPUS_LOCK:
        cached = _CORPUS_CACHE.get(collection_name)
        if cached is not None:
            return cached
        corpus = _build_corpus(collection_name)
        _CORPUS_CACHE[collection_name] = corpus
        return corpus


def _dedupe_select(
    scored: List[Tuple[float, _CorpusRecord]],
    k: int,
    max_per_title: int,
) -> List[_CorpusRecord]:
    """Select top chunks, removing near-duplicates and per-document spam."""
    selected: List[_CorpusRecord] = []
    selected_token_sets: List[Set[str]] = []
    title_counts: Counter = Counter()

    for _score, record in scored:
        if len(selected) >= k:
            break
        if title_counts[record.title] >= max_per_title:
            continue
        if any(_jaccard(record.token_set, other) > 0.85 for other in selected_token_sets):
            continue
        selected.append(record)
        selected_token_sets.append(record.token_set)
        title_counts[record.title] += 1

    return selected


def _expand_with_neighbours(
    corpus: _CorpusIndex,
    selected: List[_CorpusRecord],
    max_extra: int,
) -> List[_CorpusRecord]:
    """Append the chunks immediately before/after the best hits."""
    if max_extra <= 0:
        return selected

    seen = {record.id for record in selected}
    expanded: List[_CorpusRecord] = []
    added = 0

    for record in selected:
        expanded.append(record)
        if added >= max_extra:
            continue
        source_key = (
            str(record.metadata.get("source", "")),
            str(record.metadata.get("format", "")),
        )
        siblings = corpus.by_source.get(source_key, [])
        if len(siblings) <= 1:
            continue
        position = next(
            (i for i, sibling in enumerate(siblings) if sibling.id == record.id),
            None,
        )
        if position is None:
            continue
        # Prefer the following chunk, then the preceding one.
        for neighbour_pos in (position + 1, position - 1):
            if added >= max_extra:
                break
            if 0 <= neighbour_pos < len(siblings):
                neighbour = siblings[neighbour_pos]
                if neighbour.id not in seen:
                    seen.add(neighbour.id)
                    expanded.append(neighbour)
                    added += 1

    return expanded


class HybridRetriever(BaseRetriever):
    """Dense vector search + BM25, merged, deduplicated and context-expanded.

    Searches the main Qdrant collection plus any number of ephemeral session
    upload collections (e.g. ``zx_bank_uploads_<session_id>``). Dense search is
    run on every collection; BM25 keyword scoring is built from the main
    collection corpus only.
    """

    k: int = Field(default=8)
    fetch_k: int = Field(default=40)
    keyword_weight: float = Field(default=0.35)
    expand_neighbors: bool = Field(default=True)
    max_per_title: int = Field(default=3)
    collection_name: Optional[str] = None
    extra_collections: List[str] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        settings = get_settings()
        name = self.collection_name or settings.qdrant_collection
        client = get_qdrant_client()
        embeddings = get_embeddings()

        query_vector = embeddings.embed_query(query)
        query_tokens = _tokenize(query)

        # Collect all collections to search: main + session uploads.
        all_collections: List[str] = [name] + [
            c for c in self.extra_collections if c != name
        ]

        # Dense candidates from every collection.
        dense_by_collection: Dict[str, List] = {}
        for coll in all_collections:
            try:
                hits = client.query_points(
                    collection_name=coll,
                    query=query_vector,
                    limit=max(self.fetch_k, self.k),
                    with_payload=True,
                    with_vectors=False,
                ).points
            except Exception:
                # Collection may not exist yet.
                hits = []
            dense_by_collection[coll] = hits

        # Build corpus + BM25 from the main collection only.
        corpus = _get_corpus(name)
        bm25_scores = corpus.bm25.scores(query_tokens) if query_tokens else {}
        top_keyword = sorted(
            bm25_scores.items(), key=lambda item: item[1], reverse=True
        )[: self.fetch_k]

        # Candidate IDs from dense hits across all collections.
        candidate_ids: Set[str] = set()
        for hits in dense_by_collection.values():
            candidate_ids.update(str(hit.id) for hit in hits)
        # Candidate IDs from BM25 (main collection only).
        candidate_ids.update(corpus.records[index].id for index, _ in top_keyword)

        # Normalise dense scores per collection, then merge.
        dense_by_id: Dict[str, float] = {}
        max_dense = 0.0
        for coll, hits in dense_by_collection.items():
            for hit in hits:
                cid = str(hit.id)
                score = float(hit.score)
                dense_by_id[cid] = max(dense_by_id.get(cid, 0.0), score)
                max_dense = max(max_dense, score)
        max_dense = max_dense or 1.0

        max_keyword = max((score for _, score in top_keyword), default=0.0) or 1.0

        query_token_set = set(query_tokens)
        scored: List[Tuple[float, _CorpusRecord]] = []
        for candidate_id in candidate_ids:
            record = corpus.by_id.get(candidate_id)
            if record is None:
                continue
            dense_score = dense_by_id.get(candidate_id, 0.0) / max_dense
            keyword_score = (
                bm25_scores.get(corpus.position_by_id.get(candidate_id, -1), 0.0)
                / max_keyword
            )
            score = (
                (1.0 - self.keyword_weight) * dense_score
                + self.keyword_weight * keyword_score
            )

            # Small boosts when the query terms appear in the title/section.
            title_tokens = set(_tokenize(record.title))
            if query_token_set and title_tokens:
                title_overlap = len(query_token_set & title_tokens) / len(query_token_set)
                score += 0.15 * title_overlap
            section_tokens = set(_tokenize(record.section))
            if query_token_set and section_tokens:
                section_overlap = len(query_token_set & section_tokens) / len(query_token_set)
                score += 0.05 * section_overlap

            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)

        # Remove duplicates and cap per document.
        selected = _dedupe_select(scored, self.k, self.max_per_title)

        # Give the LLM the whole section, not an isolated fragment.
        if self.expand_neighbors:
            selected = _expand_with_neighbours(
                corpus,
                selected,
                max_extra=min(4, max(0, self.k // 2)),
            )

        return [record.as_document() for record in selected]


class DenseRetriever(BaseRetriever):
    """Dense cosine similarity search over Qdrant (brute-force via query_points)."""

    k: int = Field(default=8)
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
        return [_document_from_point(hit) for hit in result.points]


class MMRRetriever(BaseRetriever):
    """Maximal Marginal Relevance retriever.

    MMR balances relevance to the query with diversity among results. We fetch
    ``fetch_k`` candidates via dense search, then re-rank with MMR using
    cosine similarity between candidates and the query, penalizing candidates
    similar to already-selected ones.
    """

    k: int = Field(default=8)
    fetch_k: int = Field(default=30)
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
        candidates = []
        for hit in hits:
            vector = hit.vector
            if vector is None:
                continue
            candidate_vec = np.array(vector, dtype=np.float32)
            candidates.append((hit, _cosine_similarity(query_vec, candidate_vec), candidate_vec))

        selected: List = []
        remaining = list(candidates)
        while remaining and len(selected) < self.k:
            if not selected:
                best = max(remaining, key=lambda candidate: candidate[1])
            else:
                best = None
                best_score = -float("inf")
                for hit, similarity, candidate_vec in remaining:
                    max_similarity = 0.0
                    for selected_hit, _, _ in selected:
                        selected_vector = selected_hit.vector
                        if selected_vector is None:
                            continue
                        max_similarity = max(
                            max_similarity,
                            _cosine_similarity(
                                candidate_vec,
                                np.array(selected_vector, dtype=np.float32),
                            ),
                        )
                    mmr = (
                        self.lambda_mult * similarity
                        - (1 - self.lambda_mult) * max_similarity
                    )
                    if mmr > best_score:
                        best_score = mmr
                        best = (hit, similarity, candidate_vec)
                if best is None:
                    break
            selected.append(best[0])
            remaining = [c for c in remaining if c[0].id != best[0].id]

        return [_document_from_point(hit) for hit, _, _ in selected]


class MetadataFilteredRetriever(BaseRetriever):
    """Similarity search with a Qdrant metadata filter."""

    k: int = Field(default=8)
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
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in self.filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        result = client.query_points(
            collection_name=name,
            query=query_vector,
            limit=self.k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )
        return [_document_from_point(hit) for hit in result.points]


def get_retriever(
    strategy: str = "hybrid",
    k: int = 8,
    collection_name: Optional[str] = None,
    extra_collections: Optional[List[str]] = None,
    **kwargs,
) -> BaseRetriever:
    """Factory to build a retriever by strategy name.

    Args:
        strategy: One of "hybrid", "dense", "mmr", "filtered".
        k: Number of documents to return.
        collection_name: Qdrant collection name (main collection).
        extra_collections: Additional collections to search (e.g. session
            upload collections ``zx_bank_uploads_<session_id>``).
        **kwargs: Strategy-specific options (e.g. filter for "filtered",
            fetch_k/lambda_mult for "mmr", keyword_weight for "hybrid").
    """
    settings = get_settings()
    extra = extra_collections or []

    if strategy == "dense":
        return DenseRetriever(k=k, collection_name=collection_name)

    if strategy == "mmr":
        return MMRRetriever(
            k=k,
            fetch_k=kwargs.get("fetch_k", 30),
            lambda_mult=kwargs.get("lambda_mult", 0.5),
            collection_name=collection_name,
        )

    if strategy == "filtered":
        return MetadataFilteredRetriever(
            k=k,
            filter=kwargs.get("filter"),
            collection_name=collection_name,
        )

    # Default: hybrid dense + keyword retrieval.
    return HybridRetriever(
        k=k,
        fetch_k=kwargs.get("fetch_k", settings.retrieval_fetch_k),
        keyword_weight=kwargs.get(
            "keyword_weight", settings.retrieval_keyword_weight
        ),
        expand_neighbors=kwargs.get(
            "expand_neighbors", settings.retrieval_expand_neighbors
        ),
        max_per_title=kwargs.get(
            "max_per_title", settings.retrieval_max_per_title
        ),
        collection_name=collection_name,
        extra_collections=extra,
    )
