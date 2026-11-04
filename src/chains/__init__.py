"""RAG chains built with LangChain Expression Language (LCEL).

Why LCEL:
    LCEL lets us compose the RAG pipeline as a chain of runnables:
    ``RunnableParallel`` retrieves context and passes the question through,
    ``RunnablePassthrough`` forwards fields unchanged, and ``RunnableLambda``
    transforms data between steps. The result is a single ``Runnable`` that
    maps a question to a grounded answer with source citations.

Key LangChain classes used:
    * ChatOllama            - calls the qwen3:4b LLM via Ollama
    * ChatPromptTemplate    - structures system/human/AI messages
    * MessagesPlaceholder   - injects chat history for conversational RAG
    * RunnableParallel      - runs retrieval and question-passing in parallel
    * RunnablePassthrough   - passes a field through unchanged
    * RunnableLambda        - wraps plain Python functions as runnables
    * StrOutputParser        - extracts the string content from an LLM message

Why the Qwen3 settings matter:
    Qwen3 is a reasoning model. Its "thinking" tokens are generated before the
    final answer. If ``num_predict`` (output budget) is too small, the model
    spends the entire budget thinking and returns an empty answer. If thinking
    is disabled on this Ollama build, the raw chain-of-thought leaks into the
    answer text instead. We therefore keep reasoning ON and give it a generous
    context/output budget (see ``src/config/settings.py``).

Information flow (basic RAG):
    {question}
        --> HybridRetriever (dense + BM25, dedupe, neighbour expansion)
        --> format_docs_with_sources()
        --> ChatPromptTemplate (system + history + question)
        --> ChatOllama(qwen3:4b)
        --> {answer, sources}
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.config.settings import Settings, get_settings
from src.retrieval.retrievers import get_retriever
from src.prompts import get_rag_prompt


def format_docs_with_sources(docs: List[Document]) -> str:
    """Format retrieved documents into a context string with citations.

    Each chunk is prefixed with its source title, section and page/slide so the
    LLM can cite specific documents in its answer.
    """
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        title = meta.get("title", "unknown")
        section = meta.get("section")
        page = meta.get("page")
        slide = meta.get("slide")
        loc = []
        if section and section != title:
            loc.append(f'"{section}"')
        if page is not None:
            loc.append(f"p. {page}")
        if slide is not None:
            loc.append(f"slide {slide}")
        loc_str = ", ".join(loc) if loc else ""
        citation = f"{title}" + (f" ({loc_str})" if loc_str else "")
        parts.append(
            f"[Source {i+1}: {citation}]\n{doc.page_content}\n"
        )
    return "\n".join(parts)


def format_docs_plain(docs: List[Document]) -> str:
    """Format retrieved documents as plain context (no citation markup)."""
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract unique source metadata from retrieved documents for the API."""
    sources = []
    seen = set()
    for doc in docs:
        meta = doc.metadata or {}
        key = (
            meta.get("title"),
            meta.get("page"),
            meta.get("slide"),
            meta.get("section"),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": meta.get("title", "unknown"),
                "source": meta.get("source", ""),
                "format": meta.get("format", ""),
                "page": meta.get("page"),
                "slide": meta.get("slide"),
                "section": meta.get("section"),
                "chunk_index": meta.get("chunk_index"),
                "content_preview": doc.page_content[:300],
            }
        )
    return sources


def _build_llm(
    settings: Settings,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ChatOllama:
    """Create a ChatOllama client with reasoning-friendly defaults."""
    return ChatOllama(
        model=model or settings.llm_model,
        base_url=base_url or settings.ollama_base_url,
        temperature=settings.llm_temperature,
        reasoning=settings.llm_reasoning,
        num_ctx=settings.llm_num_ctx,
        num_predict=settings.llm_num_predict,
    )


def _build_retry_llm(settings: Settings) -> ChatOllama:
    """Build a new ChatOllama instance with a larger output budget for retry.

    We cannot use ``llm.bind(num_predict=..., num_ctx=...)`` because
    langchain-ollama passes bound kwargs straight to the ollama
    ``client.chat()`` method, which rejects ``num_predict`` as an
    unexpected keyword argument. Creating a fresh instance avoids that.
    """
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        reasoning=settings.llm_reasoning,
        num_ctx=max(settings.llm_num_ctx * 2, 32768),
        num_predict=max(settings.llm_num_predict * 2, 6144),
    )


class BasicRAGChain:
    """Single-turn RAG chain: question -> detailed answer with sources."""

    def __init__(
        self,
        retriever_strategy: str = "hybrid",
        k: int = 8,
        collection_name: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.retriever = get_retriever(
            strategy=retriever_strategy,
            k=k,
            collection_name=collection_name,
        )
        self.prompt = get_rag_prompt()
        self.llm = _build_llm(settings, model, base_url)
        self.chain = self._build_chain()

    def _build_chain(self):
        """Build the LCEL RAG chain."""
        return (
            RunnableParallel(
                {
                    "context": self.retriever | RunnableLambda(format_docs_with_sources),
                    "question": RunnablePassthrough(),
                    "chat_history": RunnableLambda(lambda _: []),
                }
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _build_retry_llm(self) -> ChatOllama:
        """Build a new ChatOllama instance with a larger output budget for retry."""
        return _build_retry_llm(self.settings)

    def _invoke_with_retry(self, messages) -> str:
        """Invoke the LLM; retry with a larger budget if the answer is empty."""
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = (answer or "").strip()
        if answer:
            return answer

        # Reasoning consumed the whole output budget. Retry once with more room.
        retry_llm = self._build_retry_llm()
        response = retry_llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return (answer or "").strip() or (
            "I could not produce an answer from the retrieved documents. "
            "Please try rephrasing the question."
        )

    def invoke(self, question: str) -> Tuple[str, List[Document]]:
        """Run the chain and return (answer, retrieved_documents).

        We run the retriever separately to also return the source documents
        for citation in the API layer.
        """
        docs = self.retriever.invoke(question)
        context = format_docs_with_sources(docs)
        messages = self.prompt.format(
            context=context,
            question=question,
            chat_history=[],
        )
        answer = self._invoke_with_retry(messages)
        return answer, docs


class ConversationalRAGChain:
    """Conversational RAG chain with chat history.

    Maintains a per-session message history. Uses MessagesPlaceholder in the
    prompt so prior turns are injected into the LLM context.
    """

    def __init__(
        self,
        retriever_strategy: str = "hybrid",
        k: int = 8,
        collection_name: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.retriever = get_retriever(
            strategy=retriever_strategy,
            k=k,
            collection_name=collection_name,
        )
        self.prompt = get_rag_prompt()
        self.llm = _build_llm(settings, model, base_url)

    def _build_retry_llm(self) -> ChatOllama:
        """Build a new ChatOllama instance with a larger output budget for retry."""
        return _build_retry_llm(self.settings)

    def _invoke_with_retry(self, messages) -> str:
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = (answer or "").strip()
        if answer:
            return answer

        retry_llm = self._build_retry_llm()
        response = retry_llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return (answer or "").strip() or (
            "I could not produce an answer from the retrieved documents. "
            "Please try rephrasing the question."
        )

    def invoke(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, List[Document]]:
        """Run conversational RAG.

        Args:
            question: The current user question.
            chat_history: List of (role, content) tuples for prior turns.

        Returns:
            (answer, retrieved_documents).
        """
        history = chat_history or []
        docs = self.retriever.invoke(question)
        context = format_docs_with_sources(docs)

        # Convert history tuples into LangChain message format expected by
        # MessagesPlaceholder. We pass them as a list of dicts with 'role'
        # and 'content'; ChatPromptTemplate handles conversion.
        history_messages = [
            {"role": role, "content": content} for role, content in history
        ]

        messages = self.prompt.format(
            context=context,
            question=question,
            chat_history=history_messages,
        )
        answer = self._invoke_with_retry(messages)
        return answer, docs
