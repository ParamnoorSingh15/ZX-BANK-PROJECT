"""CLI entry point to run document ingestion into Qdrant.

Usage:
    python scripts/ingest.py [--chunk-size 500] [--chunk-overlap 50]

Reads configuration from .env and indexes the ZX Bank dataset into Qdrant.
"""

from __future__ import annotations

import argparse
import sys

from src.ingestion.pipeline import run_ingestion
from src.vectorstore.qdrant_store import add_chunks
from src.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ZX Bank documents into Qdrant")
    parser.add_argument("--chunk-size", type=int, default=None, help="Max chars per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Overlap between chunks")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Dataset path: {settings.dataset_path}")
    print(f"Chunk size: {args.chunk_size or settings.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap or settings.chunk_overlap}")

    chunks = run_ingestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Loaded and split {len(chunks)} chunks")

    added = add_chunks(chunks)
    print(f"Indexed {added} new chunks into Qdrant collection '{settings.qdrant_collection}'")


if __name__ == "__main__":
    main()
