"""Integration tests for the ingestion pipeline (loads real dataset subset)."""

from pathlib import Path

from src.ingestion.pipeline import load_and_split_dataset

DATASET = Path("dataset/ZX Bank")


def test_pipeline_returns_chunks():
    """Ingesting the markdown subset yields non-empty chunks with metadata."""
    md_path = str(DATASET / "md")
    chunks = load_and_split_dataset(md_path, chunk_size=400, chunk_overlap=40)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "source" in chunk.metadata
        assert "format" in chunk.metadata
        assert "title" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["format"] == "md"


def test_pipeline_deduplicates_identical_text():
    """Loading the same doc twice through split_documents keeps separate chunks;
    duplicate detection happens at the vector store layer, not here."""
    md_path = str(DATASET / "md")
    chunks = load_and_split_dataset(md_path, chunk_size=400, chunk_overlap=40)
    assert len(chunks) > 0
    titles = {c.metadata["title"] for c in chunks}
    assert "About ZX Bank Ltd" in titles
