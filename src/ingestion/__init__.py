"""Document ingestion package: load, clean, split, and index ZX Bank documents."""

from .pipeline import run_ingestion, load_and_split_dataset

__all__ = ["run_ingestion", "load_and_split_dataset"]
