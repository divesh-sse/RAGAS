from __future__ import annotations

import json
from evals.metrics import METRIC_NAMES


def _avg(scores: list) -> float | None:
    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 3) if valid else None


def _badge(score) -> str:
    if score is None:
        return "⬜"
    if score >= 0.75:
        return "🟢"
    if score >= 0.50:
        return "🟡"
    return "🔴"


def build_results(enriched: list[dict], scores: dict) -> dict:
    """
    Returns a structured results dict:
    {
        "per_golden": [ {id, user_input, response, reference, retrieved_contexts, scores: {metric: float}} ],
        "averages":   { metric: float }
    }
    """
    per_golden = []
    for i, e in enumerate(enriched):
        per_golden.append(
            {
                "id": e["id"],
                "metric_focus": e.get("metric_focus", ""),
                "user_input": e["user_input"],
                "response": e["response"],
                "reference": e["reference"],
                "retrieved_contexts": e["retrieved_contexts"],
                "scores": {
                    name: scores.get(name, [None] * len(enriched))[i]
                    for name in METRIC_NAMES
                },
            }
        )

    averages = {name: _avg(scores.get(name, [])) for name in METRIC_NAMES}
    return {"per_golden": per_golden, "averages": averages}


def save_results(path: str, results: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(results: dict) -> None:
    print("\n" + "═" * 72)
    print("  TECHNEST RAG — EVALUATION RESULTS")
    print("═" * 72)

    header = f"  {'Question':<32}" + "".join(
        f"  {n[:8]:<9}" for n in METRIC_NAMES
    )
    print(header)
    print("  " + "─" * 68)

    for g in results["per_golden"]:
        q = g["user_input"][:30] + ".."
        row = f"  {q:<32}"
        for name in METRIC_NAMES:
            s = g["scores"].get(name)
            row += f"  {_badge(s)} {s:.2f} " if s is not None else "  ⬜ N/A "
        print(row)

    print("  " + "─" * 68)
    avg_row = f"  {'AVERAGE':<32}"
    for name in METRIC_NAMES:
        a = results["averages"].get(name)
        avg_row += f"  {_badge(a)} {a:.2f} " if a is not None else "  ⬜ N/A "
    print(avg_row)
    print("═" * 72 + "\n")
