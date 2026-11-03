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
    * StrOutputParser       - extracts the string content from an LLM message

Information flow (basic RAG):
    {question}
        --> RunnableParallel({
              context: retriever | format_docs,
              question: RunnablePassthrough(),
              chat_history: RunnablePassthrough(),
            })
        --> ChatPromptTemplate (system + history + question)
        --> ChatOllama(qwen3:4b)
        --> StrOutputParser
    --> {answer, sources}

Information flow (context construction):
    [Document(page_content, metadata)] --> format_docs()
        --> "Source: title (p. N)\ncontent\n\nSource: ..."
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

from src.config.settings import get_settings
from src.retrieval.retrievers import get_retriever
from src.prompts import get_rag_prompt


def format_docs_with_sources(docs: List[Document]) -> str:
    """Format retrieved documents into a context string with citations.

    Each chunk is prefixed with its source title and page/slide number so the
    LLM can cite specific documents in its answer.
    """
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        title = meta.get("title", "unknown")
        source = meta.get("source", "")
        page = meta.get("page")
        slide = meta.get("slide")
        loc = []
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
    """Extract source metadata from retrieved documents for API responses."""
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append(
            {
                "title": meta.get("title", "unknown"),
                "source": meta.get("source", ""),
                "format": meta.get("format", ""),
                "page": meta.get("page"),
                "slide": meta.get("slide"),
                "chunk_index": meta.get("chunk_index"),
                "content_preview": doc.page_content[:200],
            }
        )
    return sources


class BasicRAGChain:
    """Single-turn RAG chain: question -> answer with sources.

    Uses LCEL: retrieval + question pass-through in parallel, then prompt +
    LLM + output parser.
    """

    def __init__(
        self,
        retriever_strategy: str = "dense",
        k: int = 4,
        collection_name: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.retriever = get_retriever(
            strategy=retriever_strategy,
            k=k,
            collection_name=collection_name,
        )
        self.prompt = get_rag_prompt()
        self.llm = ChatOllama(
            model=model or settings.llm_model,
            base_url=base_url or settings.ollama_base_url,
            temperature=0,
        )
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
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, docs


class ConversationalRAGChain:
    """Conversational RAG chain with chat history.

    Maintains a per-session message history. Uses MessagesPlaceholder in the
    prompt so prior turns are injected into the LLM context.
    """

    def __init__(
        self,
        retriever_strategy: str = "dense",
        k: int = 4,
        collection_name: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.retriever = get_retriever(
            strategy=retriever_strategy,
            k=k,
            collection_name=collection_name,
        )
        self.prompt = get_rag_prompt()
        self.llm = ChatOllama(
            model=model or settings.llm_model,
            base_url=base_url or settings.ollama_base_url,
            temperature=0,
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
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, docs
