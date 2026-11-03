"""CLI entry point to run document ingestion into Qdrant.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --reset
    python scripts/ingest.py --chunk-size 1600 --chunk-overlap 200

Reads configuration from .env and indexes the ZX Bank dataset into Qdrant.

Use ``--reset`` whenever you change the chunk size, the embedding model or the
loaders. Without it, the old chunks stay in the collection next to the new
ones and retrieval returns a mix of stale and fresh fragments.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from src.ingestion.pipeline import run_ingestion
from src.vectorstore.qdrant_store import add_chunks, reset_collection
from src.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ZX Bank documents into Qdrant")
    parser.add_argument("--chunk-size", type=int, default=None, help="Max chars per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Overlap between chunks")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before ingesting (recommended after changing chunking)",
    )
    args = parser.parse_args()

    settings = get_settings()
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap

    print(f"Dataset path: {settings.dataset_path}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")

    if args.reset:
        deleted = reset_collection(collection_name=settings.qdrant_collection)
        print(
            f"Reset collection '{settings.qdrant_collection}': "
            f"{'deleted' if deleted else 'did not exist'}"
        )

    started = time.time()
    chunks = run_ingestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    elapsed = time.time() - started

    by_format = Counter(
        chunk.metadata.get("format", "unknown") for chunk in chunks
    )
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    small = sum(1 for chunk in chunks if len(chunk.page_content.strip()) < 120)

    print(f"Loaded and split {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"Chunks by format: {dict(by_format)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Chunks under 120 chars: {small}")

    added = add_chunks(chunks)
    print(f"Indexed {added} new chunks into Qdrant collection '{settings.qdrant_collection}'")
    print(f"Total time: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
