from src.evaluation import run_evaluation, EvalConfig

report = run_evaluation(EvalConfig(k=4, use_ragas=False))
print("Retrieval metrics:")
for k, v in report.retrieval.items():
    print(f"  {k}: {v:.3f}")
print("Generation metrics:")
for k, v in report.generation.items():
    print(f"  {k}: {v:.3f}")
print()
print("Per-question:")
for pq in report.per_question:
    ans = pq["answer"].encode("ascii", "replace").decode()
    print(f"  {pq['id']}: answer={ans[:80]}...")
