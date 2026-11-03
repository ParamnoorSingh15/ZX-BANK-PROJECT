"""Evaluation metrics and runner for ZX Bank RAG.

Why evaluation:
    RAG quality depends on both retrieval (did we find the right chunks?)
    and generation (is the answer faithful to the context?). We measure both
    so we can compare chunk sizes, retrieval strategies and prompts
    reproducibly.

Metrics:
    * Retrieval: Recall@K, Precision@K, MRR, nDCG@K
    * Generation: faithfulness, answer relevancy, correctness (via RAGAS
      where appropriate, plus deterministic heuristics)

Information flow:
    [benchmark question] + [ground-truth doc ids]
        --> retriever  --> [retrieved doc ids]
        -->  Recall@K / Precision@K / MRR / nDCG
    [question] + [retrieved context]  -->  RAG chain  --> [answer]
        -->  RAGAS / heuristics  -->  faithfulness / relevancy / correctness

Benchmark questions:
    No benchmark file existed in the project, so a small derived set is kept
    under ``data/benchmark/questions.jsonl``. Each line has:
      {"id": "...", "question": "...", "expected_doc": "...", "expected_answer": "..."}
    Expected answers come from actual document content (not fabricated).
    Results are never fabricated — only computed from live runs.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config.settings import get_settings
from src.chains import BasicRAGChain
from src.retrieval.retrievers import get_retriever
from src.vectorstore.qdrant_store import ensure_collection, get_qdrant_client
from src.embeddings.client import get_embeddings


SETTINGS = get_settings()
BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_FILE = BENCHMARK_DIR / "questions.jsonl"


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    if not retrieved_k:
        return 0.0
    return sum(1 for doc_id in retrieved_k if doc_id in relevant_set) / len(retrieved_k)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant document."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    def dcg(doc_ids: List[str]) -> float:
        total = 0.0
        for i, doc_id in enumerate(doc_ids, start=1):
            if doc_id in relevant_set:
                total += 1.0 / math.log2(i + 1)
        return total

    ideal_dcg = dcg(list(relevant_set)[:k])
    if ideal_dcg == 0:
        return 0.0
    return dcg(retrieved[:k]) / ideal_dcg


@dataclass
class RetrievalResult:
    question_id: str
    question: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    k: int
    recall: float = 0.0
    precision: float = 0.0
    mrr_score: float = 0.0
    ndcg: float = 0.0

    def compute(self, k: int) -> None:
        self.k = k
        self.recall = recall_at_k(self.retrieved_ids, self.relevant_ids, k)
        self.precision = precision_at_k(self.retrieved_ids, self.relevant_ids, k)
        self.mrr_score = mrr(self.retrieved_ids, self.relevant_ids)
        self.ndcg = ndcg_at_k(self.retrieved_ids, self.relevant_ids, k)


# ---------------------------------------------------------------------------
# Generation metrics (lightweight heuristics; RAGAS where available)
# ---------------------------------------------------------------------------

def _has_overlap(answer: str, text: str) -> bool:
    """Simple overlap check: does the answer share key terms with text?"""
    answer_words = set(w.lower().strip(".,!?:;\"'()[]") for w in answer.split())
    text_words = set(w.lower().strip(".,!?:;\"'()[]") for w in text.split())
    if not answer_words:
        return False
    overlap = answer_words & text_words
    return len(overlap) / len(answer_words) > 0.2


def faithfulness_heuristic(answer: str, context: str) -> float:
    """Heuristic faithfulness: 1.0 if answer terms largely overlap context."""
    return 1.0 if _has_overlap(answer, context) else 0.0


def answer_relevancy_heuristic(answer: str, question: str) -> float:
    """Heuristic answer relevancy: 1.0 if answer shares terms with question."""
    return 1.0 if _has_overlap(answer, question) else 0.0


def correctness_heuristic(answer: str, expected_answer: str) -> float:
    """Heuristic correctness: term overlap with expected answer."""
    if not expected_answer:
        return 0.0
    return 1.0 if _has_overlap(answer, expected_answer) else 0.0


@dataclass
class GenerationResult:
    question_id: str
    question: str
    answer: str
    context: str
    expected_answer: Optional[str]
    faithfulness: float = 0.0
    relevancy: float = 0.0
    correctness: float = 0.0

    def compute(self) -> None:
        self.faithfulness = faithfulness_heuristic(self.answer, self.context)
        self.relevancy = answer_relevancy_heuristic(self.answer, self.question)
        self.correctness = correctness_heuristic(
            self.answer, self.expected_answer or ""
        )


# ---------------------------------------------------------------------------
# RAGAS wrapper (optional, used where appropriate)
# ---------------------------------------------------------------------------

def try_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[Optional[str]],
) -> Optional[Dict[str, float]]:
    """Run RAGAS faithfulness, answer_relevancy, correctness if available.

    Returns None if RAGAS cannot be initialized, so the pipeline never
    hard-fails on evaluation.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, correctness
        from datasets import Dataset
    except Exception:
        return None

    try:
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": [gt or "" for gt in ground_truths],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, correctness],
        )
        return {
            "ragas_faithfulness": float(result["faithfulness"][0]),
            "ragas_answer_relevancy": float(result["answer_relevancy"][0]),
            "ragas_correctness": float(result["correctness"][0]),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

def load_benchmark(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load benchmark questions from JSONL.

    Each line: {"id", "question", "expected_doc", "expected_answer"}
    """
    target = Path(path) if path else BENCHMARK_FILE
    if not target.exists():
        return []
    items = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_benchmark(items: List[Dict[str, Any]], path: Optional[str] = None) -> None:
    target = Path(path) if path else BENCHMARK_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    chunk_size: int = 400
    chunk_overlap: int = 40
    retrieval_strategy: str = "dense"
    k: int = 4
    use_ragas: bool = False


@dataclass
class EvalReport:
    config: EvalConfig
    retrieval: Dict[str, float]
    generation: Dict[str, float]
    per_question: List[Dict[str, Any]] = field(default_factory=list)
    ragas: Optional[Dict[str, float]] = None


def run_evaluation(
    config: Optional[EvalConfig] = None,
    benchmark_path: Optional[str] = None,
) -> EvalReport:
    """Run the full evaluation: retrieval + generation metrics.

    Compares one configuration (chunk size, strategy, k) against the
    benchmark. Results are deterministic for a given Qdrant index.
    """
    config = config or EvalConfig()
    benchmark = load_benchmark(benchmark_path)
    if not benchmark:
        return EvalReport(
            config=config,
            retrieval={},
            generation={},
        )

    # Ensure collection exists with current settings.
    client = get_qdrant_client()
    embeddings = get_embeddings()
    ensure_collection(client, embeddings, SETTINGS.qdrant_collection)

    retriever = get_retriever(
        strategy=config.retrieval_strategy,
        k=config.k,
        collection_name=SETTINGS.qdrant_collection,
    )

    retrieval_results: List[RetrievalResult] = []
    generation_results: List[GenerationResult] = []

    for item in benchmark:
        qid = item["id"]
        question = item["question"]
        expected_doc = item.get("expected_doc", "")
        expected_answer = item.get("expected_answer", "")

        # Retrieve
        docs = retriever.invoke(question)
        retrieved_ids = [
            (d.metadata or {}).get("title", "") for d in docs if d.page_content.strip()
        ]
        relevant_ids = [expected_doc] if expected_doc else []

        rres = RetrievalResult(
            question_id=qid,
            question=question,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=config.k,
        )
        rres.compute(config.k)
        retrieval_results.append(rres)

        # Generate
        chain = BasicRAGChain(
            retriever_strategy=config.retrieval_strategy,
            k=config.k,
            collection_name=SETTINGS.qdrant_collection,
        )
        context = "\n".join(d.page_content for d in docs if d.page_content.strip())
        answer, _ = chain.invoke(question)

        gres = GenerationResult(
            question_id=qid,
            question=question,
            answer=answer,
            context=context,
            expected_answer=expected_answer,
        )
        gres.compute()
        generation_results.append(gres)

    # Aggregate retrieval metrics.
    n = len(retrieval_results)
    retrieval = {
        "recall@k": sum(r.recall for r in retrieval_results) / n if n else 0.0,
        "precision@k": sum(r.precision for r in retrieval_results) / n if n else 0.0,
        "mrr": sum(r.mrr_score for r in retrieval_results) / n if n else 0.0,
        "ndcg@k": sum(r.ndcg for r in retrieval_results) / n if n else 0.0,
    }

    # Aggregate generation metrics.
    gen_n = len(generation_results)
    generation = {
        "faithfulness": sum(g.faithfulness for g in generation_results) / gen_n if gen_n else 0.0,
        "answer_relevancy": sum(g.relevancy for g in generation_results) / gen_n if gen_n else 0.0,
        "correctness": sum(g.correctness for g in generation_results) / gen_n if gen_n else 0.0,
    }

    # Optional RAGAS.
    ragas_result = None
    if config.use_ragas:
        ragas_result = try_ragas_evaluation(
            questions=[g.question for g in generation_results],
            answers=[g.answer for g in generation_results],
            contexts=[[g.context] for g in generation_results],
            ground_truths=[g.expected_answer for g in generation_results],
        )

    return EvalReport(
        config=config,
        retrieval=retrieval,
        generation=generation,
        ragas=ragas_result,
        per_question=[
            {
                "id": r.question_id,
                "retrieval": {
                    "recall": r.recall,
                    "precision": r.precision,
                    "mrr": r.mrr_score,
                    "ndcg": r.ndcg,
                },
                "generation": {
                    "faithfulness": g.faithfulness,
                    "relevancy": g.relevancy,
                    "correctness": g.correctness,
                },
                "answer": g.answer,
            }
            for r, g in zip(retrieval_results, generation_results)
        ],
    )
