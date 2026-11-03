"""Configurable recursive text splitting.

Why a recursive splitter:
    LangChain's ``RecursiveCharacterTextSplitter`` attempts to split on
    paragraph boundaries first, then sentences, then words. This keeps
    topically related text together in the same chunk, which produces more
    coherent embeddings and better retrieval than naive fixed-size splitting.

Information flow:
    [Document(cleaned text)]  -->  split_documents()  -->  [Document(chunk)]
    Each chunk Document inherits the source metadata and gains a
    ``chunk_index`` for ordering and duplicate detection.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into overlapping chunks.

    Args:
        documents: Cleaned Documents to split.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        Chunked Documents with ``chunk_index`` added to metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Annotate each chunk with a sequential index within its source document.
    current_source = None
    index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source != current_source:
            current_source = source
            index = 0
        chunk.metadata["chunk_index"] = index
        index += 1

    return chunks
