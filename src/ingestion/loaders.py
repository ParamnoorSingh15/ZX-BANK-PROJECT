"""Format-specific document loaders for the ZX Bank dataset.

Each loader reads a single file and returns a list of LangChain ``Document``
objects. Every document carries ``page_content`` plus ``metadata`` preserving
the source path, file format, title and (where extractable) page/slide number.

Why custom loaders instead of ``langchain_community``?
    The ``langchain-community`` package is being sunset and its loader APIs are
    deprecated. We use the underlying libraries directly (pypdf, beautifulsoup4,
    docx2txt, python-pptx) and wrap the extracted text in LangChain
    ``Document`` objects. This keeps dependencies light and avoids deprecated
    APIs while still producing LangChain-native ``Document`` objects.

Information flow:
    file on disk  -->  library-specific text extraction  -->  Document(page_content, metadata)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

import pypdf
from bs4 import BeautifulSoup
import docx2txt
from pptx import Presentation


def _base_metadata(source: str, **extra) -> dict:
    """Build the common metadata dict for every document."""
    path = Path(source)
    metadata = {
        "source": str(source),
        "format": path.suffix.lower().lstrip("."),
        "title": path.stem,
    }
    metadata.update(extra)
    return metadata


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file, producing one Document per page.

    Page numbers are preserved in metadata so citations can point to a specific
    page of the source document.
    """
    documents: List[Document] = []
    with open(file_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=text,
                    metadata=_base_metadata(file_path, page=index + 1),
                )
            )
    return documents


def load_html(file_path: str) -> List[Document]:
    """Load an HTML file, extracting visible text.

    ``<script>`` and ``<style>`` blocks are stripped. Tables are rendered as
    plain text so their content is available for embedding.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return [
        Document(page_content=text, metadata=_base_metadata(file_path))
    ]


def load_docx(file_path: str) -> List[Document]:
    """Load a DOCX file as a single Document."""
    text = docx2txt.process(file_path) or ""
    return [
        Document(page_content=text, metadata=_base_metadata(file_path))
    ]


def load_pptx(file_path: str) -> List[Document]:
    """Load a PPTX file, producing one Document per slide.

    Slide numbers are preserved in metadata for citation.
    """
    documents: List[Document] = []
    presentation = Presentation(file_path)
    for index, slide in enumerate(presentation.slides):
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                slide_texts.append(shape.text_frame.text)
        text = "\n".join(slide_texts)
        documents.append(
            Document(
                page_content=text,
                metadata=_base_metadata(file_path, slide=index + 1),
            )
        )
    return documents


def load_md(file_path: str) -> List[Document]:
    """Load a Markdown file as a single Document (raw markdown text)."""
    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return [
        Document(page_content=text, metadata=_base_metadata(file_path))
    ]


# Map file extension -> loader function.
LOADER_MAP = {
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".md": load_md,
}


def load_document(file_path: str) -> List[Document]:
    """Dispatch to the correct loader based on file extension."""
    extension = Path(file_path).suffix.lower()
    loader = LOADER_MAP.get(extension)
    if loader is None:
        raise ValueError(f"Unsupported file extension: {extension}")
    return loader(file_path)


def load_dataset(dataset_path: str, extensions=None) -> List[Document]:
    """Load every supported document beneath ``dataset_path``.

    Walks the directory tree, loads each file with the matching loader, and
    aggregates all Documents. Failures on individual files are reported as
    warnings so one bad file does not abort the whole ingestion run.
    """
    if extensions is None:
        extensions = set(LOADER_MAP.keys())

    documents: List[Document] = []
    for root, _dirs, files in os.walk(dataset_path):
        for file_name in files:
            extension = Path(file_name).suffix.lower()
            if extension not in extensions:
                continue
            full_path = os.path.join(root, file_name)
            try:
                documents.extend(load_document(full_path))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"Warning: failed to load {full_path}: {exc}")
    return documents
