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

Why the prompt asks for detail:
    ZX Bank questions are usually about concrete facts — rates, fees,
    eligibility, documents, branch/ATM locations, steps. A one-line answer
    hides most of the useful information in the retrieved context. The system
    prompt therefore asks for a structured, comprehensive answer while still
    forbidding invented facts.

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

Answer the user's question using ONLY the context below. The context is
retrieved from ZX Bank documents and may contain the same information in
several file formats (PDF, HTML, DOCX, PPTX, Markdown). Merge those copies;
do not repeat the same point several times.

How to answer:
- Give a detailed, well-structured answer. Start with a direct one-or-two
  sentence answer, then add the supporting details.
- Include every relevant fact from the context: numbers, interest rates,
  fees, minimum balances, eligibility rules, required documents, steps,
  timeframes, locations and contact details.
- Use short headings, bullet points or a Markdown table when that makes the
  answer easier to read (for example, a table of interest rates by tenure).
- If the user asks how to do something, give the complete step-by-step
  process, including every channel mentioned (mobile app, NetBanking, branch,
  customer care).
- If the context contains partial information, give everything that is
  available and clearly state what is missing. Never answer with only
  "the context does not specify" when some relevant details exist.
- If the context truly has no information on the question, say that ZX Bank's
  documents do not contain it and suggest contacting ZX Bank customer care.
- Do not invent rates, fees, account numbers, policies or eligibility rules
  that are not in the context.
- The user may write in English or another language; answer in the same
  language as the question.

Citations:
- Cite the document title, and the page, slide or section when available,
  in parentheses, e.g. (ZX Bank Fixed Deposits, slide 6) or
  (Savings Account Overview, "Minimum Balance Requirements").
- Cite the source next to the fact it supports, not only at the end.

Context:
{context}
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
