"""Embeddings package: Ollama-backed embedding generation for ZX Bank RAG."""

from .client import get_embeddings, probe_dimension

__all__ = ["get_embeddings", "probe_dimension"]
