"""Ollama embedding client for ZX Bank RAG.

Why this wrapper:
    Indexing and querying must use the *same* embedding model and vector
    dimensionality. Rather than hardcoding an assumed dimension (Qwen3-Embedding
    4B is typically 2560 but should not be assumed), we probe the live Ollama
    model at startup to discover the actual output dimension, then configure
    the Qdrant collection accordingly.

Information flow:
    settings (model name)  -->  OllamaEmbeddings  -->  embed_query(text)
    -->  dimension probe  -->  Qdrant collection vector size
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings


def get_embeddings(model: str | None = None, base_url: str | None = None) -> OllamaEmbeddings:
    """Instantiate an OllamaEmbeddings client.

    Args:
        model: Ollama embedding model name (defaults to settings).
        base_url: Ollama API base URL (defaults to settings).

    Returns:
        Configured ``OllamaEmbeddings`` instance.
    """
    from ..config.settings import get_settings

    settings = get_settings()
    return OllamaEmbeddings(
        model=model or settings.embedding_model,
        base_url=base_url or settings.ollama_base_url,
    )


def probe_dimension(embeddings: OllamaEmbeddings | None = None) -> int:
    """Probe the live embedding model to discover its output dimension.

    We embed a short probe string and measure the vector length. This is the
    authoritative dimension used to create the Qdrant collection, guaranteeing
    indexing and query vectors match.

    Returns:
        The embedding vector dimension as an integer.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    vector = embeddings.embed_query("dimension probe")
    return len(vector)
