"""Evaluation package: retrieval + generation metrics for ZX Bank RAG."""

from .metrics import (
    EvalConfig,
    EvalReport,
    run_evaluation,
    load_benchmark,
    save_benchmark,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    faithfulness_heuristic,
    answer_relevancy_heuristic,
    correctness_heuristic,
    try_ragas_evaluation,
)

__all__ = [
    "EvalConfig",
    "EvalReport",
    "run_evaluation",
    "load_benchmark",
    "save_benchmark",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness_heuristic",
    "answer_relevancy_heuristic",
    "correctness_heuristic",
    "try_ragas_evaluation",
]
