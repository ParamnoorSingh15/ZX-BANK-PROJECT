"""Prompt templates for ZX Bank RAG.

Why ChatPromptTemplate:
    It structures the conversation as a list of messages (system, human, AI)
    rather than a single raw string. This matches how chat models like
    Qwen3 expect input and lets us inject context, sources and chat history
    cleanly.

Why MessagesPlaceholder:
    It reserves a slot in the prompt where a variable-length list of messages
    (e.g. prior conversation turns) is inserted. Essential for conversational
    RAG where we append previous user/assistant turns.

Information flow:
    [user question] + [retrieved context] + [chat history]
        --> ChatPromptTemplate.format()  --> [messages]
        --> ChatOllama.invoke(messages)  --> [grounded answer]
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Basic RAG prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are Zia, the AI banking assistant for ZX Bank.

Answer the user's question using ONLY the provided context from ZX Bank
documents. If the answer cannot be determined from the context, say so
clearly rather than guessing.

When you use information from a specific document, cite it by mentioning
the document title and page/slide number (when available) in parentheses,
like this: (About ZX Bank Ltd., p. 1).

Context:
{context}

Guidelines:
- Be accurate, concise, and helpful.
- Do not invent account numbers, rates, or policies not present in the context.
- If the user asks about something outside ZX Bank's documented services, say you don't have that information.
- Always ground your answer in the retrieved context.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# A simpler PromptTemplate variant (no chat history) for single-turn QA.
SIMPLE_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_SYSTEM_PROMPT + "\n\nQuestion: {question}\n\nAnswer:",
)

# ---------------------------------------------------------------------------
# Query rewriting prompt (advanced RAG)
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query-rewriting assistant for a banking knowledge base. "
            "Rewrite the user's question to be clearer, more specific, and better "
            "suited for semantic search over ZX Bank documents. "
            "Return ONLY the rewritten question, no explanation.",
        ),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Multi-query prompt (advanced RAG)
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query-expansion assistant. Given a user question about "
            "ZX Bank, generate 3 alternative phrasings that would help retrieve "
            "relevant documents from a vector database. Return one phrasing per "
            "line, no numbering, no explanation.",
        ),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Answer grading / faithfulness prompt (evaluation support)
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an evaluation judge. Given an ANSWER and its supporting "
            "CONTEXT, determine whether the answer is fully supported by the "
            "context. Respond with exactly one word: YES or NO.",
        ),
        (
            "human",
            "CONTEXT:\n{context}\n\nANSWER:\n{answer}",
        ),
    ]
)


def get_rag_prompt() -> ChatPromptTemplate:
    """Return the default conversational RAG prompt."""
    return RAG_PROMPT


def get_simple_rag_prompt() -> PromptTemplate:
    """Return the single-turn RAG prompt (no chat history)."""
    return SIMPLE_RAG_PROMPT
