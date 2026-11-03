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

Why extract *everything*:
    ZX Bank documents store a large share of their facts in tables (FD rates,
    branch/ATM lists, eligibility grids). Dropping table shapes or flattening
    them badly means the retriever can find a section title but never the
    numbers underneath it. These loaders therefore walk nested PowerPoint
    shapes, tables, charts and speaker notes, and render HTML tables row by row
    so every cell's text survives into the vector store.

Information flow:
    file on disk  -->  library-specific text extraction  -->  Document(page_content, metadata)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List

from langchain_core.documents import Document

import pypdf
from bs4 import BeautifulSoup
import docx2txt
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


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
    page of the source document. If normal extraction returns nothing for a
    page (common with table-heavy PDFs), we retry in layout mode so tabular
    text is not silently lost.
    """
    documents: List[Document] = []
    with open(file_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text.strip():
                try:
                    text = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    text = ""
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
    pipe-separated rows *before* text extraction so cells stay associated with
    their row instead of becoming a loose column of words.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Replace each table with a readable row-by-row representation.
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"])
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")

    text = soup.get_text(separator="\n")
    return [
        Document(page_content=text, metadata=_base_metadata(file_path))
    ]


def load_docx(file_path: str) -> List[Document]:
    """Load a DOCX file as a single Document.

    ``docx2txt`` walks the document XML, so paragraph text and table cell text
    (including FD rate tables) are both captured.
    """
    text = docx2txt.process(file_path) or ""
    return [
        Document(page_content=text, metadata=_base_metadata(file_path))
    ]


def _iter_shapes(shapes) -> Iterator:
    """Yield every shape, descending into grouped shapes."""
    for shape in shapes:
        try:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from _iter_shapes(shape.shapes)
                continue
        except Exception:
            pass
        yield shape


def _table_text(table) -> str:
    """Render a PowerPoint table as one pipe-separated line per row."""
    rows: List[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _chart_text(chart) -> str:
    """Extract labels and values from a PowerPoint chart, if present."""
    parts: List[str] = []
    try:
        if chart.has_title and chart.chart_title.has_text_frame:
            title = chart.chart_title.text_frame.text.strip()
            if title:
                parts.append(title)
    except Exception:
        pass
    try:
        categories = list(chart.plots[0].categories)
        if categories:
            parts.append("Categories: " + " | ".join(str(c) for c in categories))
    except Exception:
        pass
    try:
        for series in chart.series:
            values = [str(v) for v in series.values if v is not None]
            if values:
                name = getattr(series, "name", "") or "Series"
                parts.append(f"{name}: " + " | ".join(values))
    except Exception:
        pass
    return "\n".join(parts)


def load_pptx(file_path: str) -> List[Document]:
    """Load a PPTX file, producing one Document per slide.

    Slide numbers are preserved in metadata for citation.

    Why we handle tables, groups, charts and notes explicitly:
        Many ZX Bank PPTX files store their data (FD rate tables, branch lists,
        ATM locations, IFSC codes, etc.) in PowerPoint *table* shapes rather
        than plain text frames. The original code only checked
        ``shape.has_text_frame``, so all table content was silently dropped —
        causing the RAG system to say "I don't have specific information" even
        when the data existed in the deck. We now capture text frames, nested
        group shapes, tables, charts and speaker notes so every word is
        available for embedding and retrieval.
    """
    documents: List[Document] = []
    presentation = Presentation(file_path)
    for index, slide in enumerate(presentation.slides):
        slide_texts: List[str] = []
        for shape in _iter_shapes(slide.shapes):
            try:
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()
                    if text:
                        slide_texts.append(text)
                elif getattr(shape, "has_table", False):
                    text = _table_text(shape.table)
                    if text:
                        slide_texts.append(text)
                elif getattr(shape, "has_chart", False):
                    text = _chart_text(shape.chart)
                    if text:
                        slide_texts.append(text)
            except Exception:
                # A single malformed shape must not abort the whole deck.
                continue

        # Speaker notes often contain the explanation behind a table.
        try:
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    slide_texts.append("Speaker notes:\n" + notes)
        except Exception:
            pass

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
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
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
