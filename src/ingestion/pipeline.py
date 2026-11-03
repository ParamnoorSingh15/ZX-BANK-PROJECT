"""End-to-end ingestion pipeline: load -> clean -> split.

This module ties the loaders, cleaners and splitter together into a single
callable so the rest of the application (and tests) can ingest a dataset with
one function call.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from .loaders import load_dataset
from .cleaners import clean_documents
from .splitter import split_documents


def load_and_split_dataset(
    dataset_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Load, clean and split every document in ``dataset_path``.

    Chunk size/overlap fall back to the env-driven settings when not given, so
    callers never silently use a different chunking than the API expects.

    Returns the list of chunk Documents ready for embedding and indexing.
    """
    from ..config.settings import get_settings

    settings = get_settings()
    documents = load_dataset(dataset_path)
    documents = clean_documents(documents)
    chunks = split_documents(
        documents,
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )
    return chunks


def run_ingestion(
    dataset_path: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Run ingestion using project settings (env-driven) unless overridden.

    This is the entry point used by the CLI/scripts and tests.
    """
    from ..config.settings import get_settings

    settings = get_settings()
    path = dataset_path or settings.dataset_path
    chunks = load_and_split_dataset(
        path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks
