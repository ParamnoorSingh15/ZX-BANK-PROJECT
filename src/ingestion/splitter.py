"""Configurable recursive text splitting.

Why a recursive splitter:
    LangChain's ``RecursiveCharacterTextSplitter`` attempts to split on
    paragraph boundaries first, then sentences, then words. This keeps
    topically related text together in the same chunk, which produces more
    coherent embeddings and better retrieval than naive fixed-size splitting.

Why we merge tiny chunks:
    Headings, slide titles and table captions often end up as chunks of a few
    characters. Those chunks embed strongly against a question that mentions
    the heading while carrying none of the facts underneath it, so the LLM
    receives a title and answers "the context does not specify". We merge any
    chunk below ``min_chunk_size`` into its neighbour before indexing.

Why we add a context header:
    Each chunk is prefixed with ``Document: <title>`` and ``Section: <heading>``.
    This keeps document identity in the embedded text (helping questions like
    "where is ZX Bank headquartered?" pick the About document over a branch
    page) and gives the LLM a natural citation label.

Information flow:
    [Document(cleaned text)]  -->  split_documents()  -->  [Document(chunk)]
    Each chunk Document inherits the source metadata and gains
    ``chunk_index``, ``chunk_total`` and ``section`` for ordering, citation and
    neighbour expansion at retrieval time.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def _find_section(text: str) -> str | None:
    """Return the first Markdown heading in a chunk, if there is one.

    The first heading is the section the chunk starts in; later headings may
    belong to following sections that happen to share the same chunk.
    """
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    return None


def _merge_tiny_chunks(
    chunks: List[Document],
    min_chunk_size: int,
    max_chunk_size: int,
) -> List[Document]:
    """Merge undersized chunks into a neighbour without crossing documents."""
    if not chunks:
        return []

    merged: List[Document] = []
    for chunk in chunks:
        current = Document(
            page_content=chunk.page_content,
            metadata=dict(chunk.metadata),
        )
        if (
            merged
            and len(merged[-1].page_content.strip()) < min_chunk_size
            and len(merged[-1].page_content) + len(current.page_content) + 1
            <= max_chunk_size
        ):
            merged[-1].page_content = (
                merged[-1].page_content.rstrip() + "\n" + current.page_content.lstrip()
            )
        else:
            merged.append(current)

    # A short trailing chunk (footer/copyright) should join the previous chunk.
    if (
        len(merged) > 1
        and len(merged[-1].page_content.strip()) < min_chunk_size
        and len(merged[-1].page_content) + len(merged[-2].page_content) + 1
        <= max_chunk_size
    ):
        last = merged.pop()
        merged[-1].page_content = (
            merged[-1].page_content.rstrip() + "\n" + last.page_content.lstrip()
        )

    return merged


def split_documents(
    documents: List[Document],
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
    min_chunk_size: int = 200,
    add_context_header: bool = True,
) -> List[Document]:
    """Split documents into overlapping, context-rich chunks.

    Args:
        documents: Cleaned Documents to split.
        chunk_size: Target max characters per chunk (before the context header).
        chunk_overlap: Characters of overlap between consecutive chunks.
        min_chunk_size: Chunks smaller than this are merged into a neighbour.
        add_context_header: Prefix each chunk with its document title and
            section heading so the text embeds with its source context.

    Returns:
        Chunked Documents with ``chunk_index``, ``chunk_total`` and ``section``
        added to metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
    )

    all_chunks: List[Document] = []
    for document in documents:
        raw_chunks = splitter.split_documents([document])
        merged = _merge_tiny_chunks(
            raw_chunks,
            min_chunk_size=min_chunk_size,
            max_chunk_size=int(chunk_size * 1.4),
        )

        title = str(document.metadata.get("title", "unknown"))
        for chunk in merged:
            section = _find_section(chunk.page_content) or title
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata["section"] = section
            if add_context_header:
                header = f"Document: {title}\nSection: {section}\n\n"
                chunk.page_content = header + chunk.page_content

        all_chunks.extend(merged)

    # Assign chunk indices per source document (across pages/slides).
    counters: Dict[Tuple[str, str], int] = {}
    for chunk in all_chunks:
        key = (
            str(chunk.metadata.get("source", "")),
            str(chunk.metadata.get("format", "")),
        )
        chunk.metadata["chunk_index"] = counters.get(key, 0)
        counters[key] = chunk.metadata["chunk_index"] + 1

    for chunk in all_chunks:
        key = (
            str(chunk.metadata.get("source", "")),
            str(chunk.metadata.get("format", "")),
        )
        chunk.metadata["chunk_total"] = counters[key]

    return all_chunks
