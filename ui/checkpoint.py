from __future__ import annotations

import json
import os

CHECKPOINT_PATH = "checkpoint.json"
RESULTS_PATH = "results.json"


def save_checkpoint(enriched, scores, errors, phase1_done, phase2_done):
    data = {
        "enriched": enriched,
        "scores": scores,
        "errors": errors,
        "phase1_done": phase1_done,
        "phase2_done": phase2_done,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_checkpoint() -> dict | None:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
