"""Text extraction and cleaning for loaded documents.

Why cleaning matters:
    Raw extraction from PDFs/HTML/PPTX often contains stray whitespace, blank
    lines, and artifacts that pollute embeddings and hurt retrieval quality.
    Cleaning normalises the text before splitting and embedding so that
    semantically identical content produces similar vectors.

Information flow:
    Document(raw text)  -->  clean_text()  -->  Document(cleaned text)
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Normalise whitespace and strip line-level clutter from a text block."""
    # Collapse 3+ newlines into a single blank line (paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs into a single space.
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace on each line.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """Apply ``clean_text`` to every document, preserving metadata."""
    cleaned: List[Document] = []
    for document in documents:
        cleaned.append(
            Document(
                page_content=clean_text(document.page_content),
                metadata=dict(document.metadata),
            )
        )
    return cleaned
