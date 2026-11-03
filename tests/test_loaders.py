"""Unit tests for document loaders."""

from pathlib import Path

from src.ingestion.loaders import load_document, LOADER_MAP
from src.ingestion.cleaners import clean_text, clean_documents
from src.ingestion.splitter import split_documents
from langchain_core.documents import Document

DATASET = Path("dataset/ZX Bank")


def test_loader_map_covers_supported_extensions():
    """Every format we claim to support has a loader registered."""
    for ext in (".pdf", ".html", ".htm", ".docx", ".pptx", ".md"):
        assert ext in LOADER_MAP


def test_load_md_returns_document_with_metadata():
    """Markdown loader returns a Document with source/title metadata."""
    path = str(DATASET / "md" / "About ZX Bank Ltd.md")
    docs = load_document(path)
    assert len(docs) == 1
    assert docs[0].page_content.startswith("# ZX Bank Ltd.")
    assert docs[0].metadata["source"] == path
    assert docs[0].metadata["format"] == "md"
    assert docs[0].metadata["title"] == "About ZX Bank Ltd"


def test_load_html_strips_tags():
    """HTML loader strips markup and returns visible text."""
    path = str(DATASET / "html" / "About ZX Bank Ltd.html")
    docs = load_document(path)
    assert len(docs) == 1
    assert "<h1" not in docs[0].page_content
    assert "ZX Bank Ltd." in docs[0].page_content
    assert docs[0].metadata["format"] == "html"


def test_clean_text_normalises_whitespace():
    """Cleaning collapses excessive whitespace and blank lines."""
    raw = "Hello   world\n\n\n\n  spaced  "
    assert clean_text(raw) == "Hello world\n\nspaced"


def test_clean_documents_preserves_metadata():
    """clean_documents keeps metadata intact."""
    docs = [Document(page_content="  a   b  ", metadata={"source": "x.md", "format": "md"})]
    cleaned = clean_documents(docs)
    assert cleaned[0].page_content == "a b"
    assert cleaned[0].metadata == {"source": "x.md", "format": "md"}


def test_split_documents_adds_chunk_index():
    """Splitting produces chunks with sequential chunk_index metadata."""
    docs = [
        Document(
            page_content="A" * 1200,
            metadata={"source": "x.md", "format": "md"},
        )
    ]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # Every chunk should carry a chunk_index.
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source"] == "x.md"
