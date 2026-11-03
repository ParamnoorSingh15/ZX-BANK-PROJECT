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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Runtime configuration for the ZX Bank RAG pipeline."""

    def __init__(self) -> None:
        # Ollama
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm_model = os.getenv("LLM_MODEL", "qwen3:4b")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")

        # Qwen3 is a reasoning model. Thinking is kept ON because this Ollama
        # build leaks raw chain-of-thought into ``content`` when it is disabled.
        # ``num_ctx`` and ``num_predict`` must be generous: if the reasoning
        # consumes the whole budget the final answer comes back empty.
        self.llm_num_ctx = _env_int("LLM_NUM_CTX", 16384)
        self.llm_num_predict = _env_int("LLM_NUM_PREDICT", 3072)
        self.llm_temperature = _env_float("LLM_TEMPERATURE", 0.3)
        self.llm_reasoning = _env_bool("LLM_REASONING", True)

        # Qdrant
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        # Trim trailing slash so collection names are clean.
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "zx_bank_knowledge").rstrip("/")

        # Dataset
        self.dataset_path = os.getenv("DATASET_PATH", "dataset/ZX Bank")

        # Chunking (configurable via env). Larger chunks keep a heading and its
        # table together, which is essential for rate/eligibility questions.
        self.chunk_size = _env_int("CHUNK_SIZE", 1600)
        self.chunk_overlap = _env_int("CHUNK_OVERLAP", 200)

        # Retrieval
        self.retrieval_strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
        self.retrieval_k = _env_int("RETRIEVAL_K", 8)
        self.retrieval_fetch_k = _env_int("RETRIEVAL_FETCH_K", 40)
        self.retrieval_keyword_weight = _env_float("RETRIEVAL_KEYWORD_WEIGHT", 0.35)
        self.retrieval_expand_neighbors = _env_bool("RETRIEVAL_EXPAND_NEIGHBORS", True)
        self.retrieval_max_per_title = _env_int("RETRIEVAL_MAX_PER_TITLE", 3)

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
