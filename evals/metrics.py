from __future__ import annotations

"""
RAGAS evaluation with safe rate-limit handling.

Rate limits (Groq free/on_demand tier):
  llama-3.1-8b-instant : 6,000 TPM  |  30 RPM  |  500,000 TPD

Token budget per sample (approx):
  Faithfulness       : 2 calls × ~550 tok  = ~1,100 tok/sample
  Answer Relevancy   : 1 call  × ~250 tok  = ~250  tok/sample
  Context Precision  : 2 calls × ~350 tok  = ~700  tok/sample
  Context Recall     : 2 calls × ~350 tok  = ~700  tok/sample
  Answer Correctness : 3 calls × ~350 tok  = ~1,050 tok/sample

Strategy: 1 sample at a time, SAMPLE_COOLDOWN between samples,
EXPERIMENT_COOLDOWN between experiments. On 429 → wait RETRY_WAIT and retry once.
"""

import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "llama-3.1-8b-instant"

# ── Timing constants ──────────────────────────────────────────────────────────
SAMPLE_COOLDOWN = 25      # seconds between samples within one experiment
EXPERIMENT_COOLDOWN = 35  # seconds between experiments (fully resets TPM window)
RETRY_WAIT = 65           # seconds to wait after a 429 before retrying

# ── Context truncation (prevents token overflow per call) ─────────────────────
CONTEXT_CHARS = 400       # max chars per context chunk passed to metrics
CONTEXT_LIMIT = 2         # max number of context chunks passed to metrics


# ── Setup helpers ─────────────────────────────────────────────────────────────

def build_judge(api_key: str) -> object:
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return llm_factory(JUDGE_MODEL, provider="openai", client=client)


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )


# ── Input preparation ─────────────────────────────────────────────────────────

def _truncate_contexts(contexts: list[str]) -> list[str]:
    return [c[:CONTEXT_CHARS] for c in contexts[:CONTEXT_LIMIT]]


def prepare_inputs(enriched: list[dict], keys: list[str]) -> list[dict]:
    result = []
    for e in enriched:
        d = {}
        for k in keys:
            if k == "retrieved_contexts":
                d[k] = _truncate_contexts(e.get("retrieved_contexts", []))
            else:
                d[k] = e.get(k, "")
        result.append(d)
    return result


# ── Scoring ───────────────────────────────────────────────────────────────────

async def _score_one(
    metric,
    input_dict: dict,
    error_collector: list | None = None,
    sample_idx: int = 0,
) -> float | None:
    """
    Score a single sample. Returns None on failure.
    Appends error dicts to error_collector (a plain list) — never calls Streamlit,
    so it is safe to run inside a ThreadPoolExecutor worker thread.
    """
    try:
        results = await metric.abatch_score([input_dict])
        return float(results[0].value)

    except Exception as e:
        err_str = str(e)
        is_rate_limit = (
            "429" in err_str
            or "rate" in err_str.lower()
            or "limit" in err_str.lower()
            or "quota" in err_str.lower()
        )

        if is_rate_limit:
            if error_collector is not None:
                error_collector.append({"sample": sample_idx + 1, "error": f"Rate limit — waiting {RETRY_WAIT}s then retrying…"})
            await asyncio.sleep(RETRY_WAIT)
            try:
                results = await metric.abatch_score([input_dict])
                return float(results[0].value)
            except Exception as e2:
                if error_collector is not None:
                    error_collector.append({"sample": sample_idx + 1, "error": f"Retry failed: {str(e2)[:120]}"})
                return None

        if error_collector is not None:
            error_collector.append({"sample": sample_idx + 1, "error": f"{type(e).__name__}: {err_str[:120]}"})
        return None


async def score_experiment(
    metric,
    inputs: list[dict],
    error_collector: list | None = None,
) -> list[float | None]:
    """
    Score one metric across all samples, one at a time with SAMPLE_COOLDOWN between each.
    error_collector is a plain list that gets appended to on failure — no Streamlit calls,
    safe to run inside a worker thread.
    """
    scores = []
    for i, inp in enumerate(inputs):
        score = await _score_one(metric, inp, error_collector=error_collector, sample_idx=i)
        scores.append(score)
        if i < len(inputs) - 1:
            await asyncio.sleep(SAMPLE_COOLDOWN)
    return scores


# ── Experiment registry ───────────────────────────────────────────────────────
# Each entry: (display_name, metric_factory(llm, emb), required_input_keys)

EXPERIMENTS = [
    (
        "Faithfulness",
        lambda llm, emb: Faithfulness(llm=llm),
        ["user_input", "response", "retrieved_contexts"],
    ),
    (
        "Answer Relevancy",
        lambda llm, emb: AnswerRelevancy(llm=llm, embeddings=emb),
        ["user_input", "response"],
    ),
    (
        "Context Precision",
        lambda llm, emb: ContextPrecision(llm=llm),
        ["user_input", "reference", "retrieved_contexts"],
    ),
    (
        "Context Recall",
        lambda llm, emb: ContextRecall(llm=llm),
        ["user_input", "retrieved_contexts", "reference"],
    ),
    (
        "Answer Correctness",
        lambda llm, emb: AnswerCorrectness(llm=llm, embeddings=emb),
        ["user_input", "response", "reference"],
    ),
]

METRIC_NAMES = [name for name, _, _ in EXPERIMENTS]
