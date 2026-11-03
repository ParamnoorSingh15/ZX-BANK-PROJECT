"""Application configuration loaded from environment variables.

All settings are read from the environment (via .env). Nothing is hardcoded
except sensible defaults. The embedding dimension is detected at runtime from
the Ollama embedding model rather than assumed, so indexing and querying always
use the same vector size.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (wherever settings.py is imported from).
load_dotenv()


class Settings:
    """Runtime configuration for the ZX Bank RAG pipeline."""

    def __init__(self) -> None:
        # Ollama
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm_model = os.getenv("LLM_MODEL", "qwen3:4b")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")

        # Qdrant
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        # Trim trailing slash so collection names are clean.
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "zx_bank_knowledge").rstrip("/")

        # Dataset
        self.dataset_path = os.getenv("DATASET_PATH", "dataset/ZX Bank")

        # Chunking (configurable via env)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

        # Detection limit for embedding dimension probe
        self.embedding_dimension: int | None = None


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def resolve_dataset_path() -> Path:
    """Resolve the dataset path relative to the project root if needed."""
    settings = get_settings()
    path = Path(settings.dataset_path)
    if not path.is_absolute():
        # Resolve relative to the project root (parent of src/)
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / path
        if candidate.exists():
            return candidate
    return path
