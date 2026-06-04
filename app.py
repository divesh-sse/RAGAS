"""
TechNest RAG Evaluation Pipeline
Run with:  streamlit run app.py
"""

import asyncio
import json
import os
import time

import nest_asyncio
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evals.metrics import (
    EXPERIMENT_COOLDOWN,
    EXPERIMENTS,
    METRIC_NAMES,
    SAMPLE_COOLDOWN,
    build_embeddings,
    build_judge,
    prepare_inputs,
    score_experiment,
)
from evals.reporter import _avg, _badge, build_results, save_results
from evals.runner import run_phase1
from rag.generator import Generator
from rag.loader import load_catalog, load_goldens
from rag.retriever import Retriever

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# nest_asyncio must be applied before any event loop is created.
# This makes asyncio.run() work inside Streamlit's own event loop on all
# platforms including Streamlit Cloud (Linux) and Windows.
nest_asyncio.apply()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JUDGE_GROQ = os.getenv("JUDGE_GROQ", GROQ_API_KEY)
CHECKPOINT_PATH = "checkpoint.json"
RESULTS_PATH = "results.json"


# ── FIX 1: Reliable async runner ──────────────────────────────────────────────
# The old approach (asyncio.get_event_loop().run_until_complete) breaks on
# Streamlit Cloud because Streamlit may own the event loop in that thread.
# asyncio.run() always creates a fresh loop and is safe with nest_asyncio applied.
def _run(coro):
    return asyncio.run(coro)


# ── FIX 2: Checkpoint helpers ─────────────────────────────────────────────────
# We save progress after every experiment so that if the connection drops
# mid-run (common on Streamlit Cloud with 15-min eval runs), nothing is lost.

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


# ── Cached heavy objects ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading catalog and building embeddings…")
def get_retriever():
    catalog = load_catalog()
    return Retriever(catalog), catalog


@st.cache_resource(show_spinner="Loading RAGAS embeddings (sentence-transformers)…")
def get_ragas_embeddings():
    return build_embeddings()


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNest RAG Evaluator",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 TechNest — RAG Evaluation Pipeline")
st.caption(
    "Build a small RAG system over a product catalog, then evaluate it "
    "with **RAGAS 0.4.3** across 5 metrics."
)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "enriched": None,
    "scores": {},        # dict keyed by experiment name, filled incrementally
    "errors": {},        # dict keyed by experiment name → list of error dicts
    "results": None,
    "phase1_done": False,
    "phase2_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── FIX 3: Sidebar — checkpoint restore ───────────────────────────────────────
# If the browser disconnected mid-run, the checkpoint file on disk still has
# the partial results. The user can restore without re-running from scratch.
with st.sidebar:
    st.header("Session")

    checkpoint = load_checkpoint()
    if checkpoint and not st.session_state.phase1_done:
        completed = list(checkpoint.get("scores", {}).keys())
        st.info(
            f"💾 Checkpoint found.\n\n"
            f"Phase 1: {'✅' if checkpoint.get('phase1_done') else '❌'}\n\n"
            f"Experiments done: {len(completed)}/5"
            + (f" ({', '.join(completed)})" if completed else "")
        )
        if st.button("🔄 Restore from checkpoint", use_container_width=True):
            st.session_state.enriched = checkpoint.get("enriched")
            st.session_state.scores = checkpoint.get("scores", {})
            st.session_state.errors = checkpoint.get("errors", {})
            st.session_state.phase1_done = checkpoint.get("phase1_done", False)
            st.session_state.phase2_done = checkpoint.get("phase2_done", False)
            if st.session_state.phase2_done and st.session_state.enriched:
                st.session_state.results = build_results(
                    st.session_state.enriched,
                    st.session_state.scores,
                )
            st.rerun()

    if st.session_state.phase1_done or st.session_state.phase2_done:
        st.success(
            f"Phase 1: {'✅' if st.session_state.phase1_done else '⏳'}\n\n"
            f"Phase 2: {'✅' if st.session_state.phase2_done else f'{len(st.session_state.scores)}/5 experiments'}"
        )

    if checkpoint and st.button("🗑️ Clear checkpoint", use_container_width=True):
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
        for key in ["enriched", "scores", "errors", "results", "phase1_done", "phase2_done"]:
            st.session_state[key] = {} if key in ("scores", "errors") else (None if key not in ("phase1_done", "phase2_done") else False)
        st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_catalog, tab_goldens, tab_pipeline, tab_results = st.tabs(
    ["📚 Catalog", "🎯 Goldens", "🚀 Run Evaluation", "📊 Results"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CATALOG
# ─────────────────────────────────────────────────────────────────────────────
with tab_catalog:
    st.subheader("Knowledge Base — TechNest Product Catalog")
    st.write(
        "15 entries across **products**, **policies**, and **FAQs** "
        "that the RAG system retrieves from."
    )

    retriever, catalog = get_retriever()
    df_cat = pd.DataFrame(catalog)[["id", "category", "title", "content"]]
    df_cat.columns = ["ID", "Category", "Title", "Content"]

    category_filter = st.multiselect(
        "Filter by category",
        options=["product", "policy", "faq"],
        default=["product", "policy", "faq"],
    )
    filtered = df_cat[df_cat["Category"].isin(category_filter)]
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.caption(f"{len(filtered)} of {len(df_cat)} entries shown")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GOLDENS
# ─────────────────────────────────────────────────────────────────────────────
with tab_goldens:
    st.subheader("Golden Dataset — 5 Q&A Pairs with Ground Truth")
    st.write(
        "Each golden targets a specific RAGAS metric so we can verify "
        "the evaluator catches real failure modes."
    )

    goldens = load_goldens()
    metric_colors = {
        "faithfulness": "🟥",
        "answer_relevancy": "🟧",
        "context_precision": "🟨",
        "context_recall": "🟩",
        "answer_correctness": "🟦",
    }

    for g in goldens:
        badge = metric_colors.get(g["metric_focus"], "⬜")
        with st.expander(f"{badge} **{g['id']}** — {g['user_input']}"):
            col1, col2 = st.columns(2)
            col1.markdown("**Metric focus**")
            col1.code(g["metric_focus"])
            col2.markdown("**Reference (ground truth)**")
            col2.info(g["reference"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    st.subheader("Evaluation Pipeline")

    if not GROQ_API_KEY:
        st.error("⚠️  GROQ_API_KEY not found in .env — please add it and restart.")
        st.stop()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    st.markdown("### Phase 1 — RAG Pipeline")
    st.write(
        "Runs the RAG system on each golden question: "
        "retrieves context chunks, then generates an answer with Groq."
    )

    if st.button("▶ Run Phase 1 — Generate RAG Responses", type="primary"):
        retriever, catalog = get_retriever()
        generator = Generator(api_key=GROQ_API_KEY)
        goldens = load_goldens()

        progress = st.progress(0, text="Starting…")
        log = st.empty()
        lines = []

        def phase1_cb(i, question):
            lines.append(f"[{i+1}/{len(goldens)}] {question[:60]}…")
            log.code("\n".join(lines))
            progress.progress((i + 1) / len(goldens), text=f"Processing {i+1}/{len(goldens)}…")

        with st.spinner("Running RAG on 5 goldens (5s spacing between calls)…"):
            enriched = _run(
                run_phase1(goldens, retriever, generator, status_callback=phase1_cb)
            )

        progress.progress(1.0, text="Phase 1 complete ✅")
        st.session_state.enriched = enriched
        st.session_state.phase1_done = True

        # Save checkpoint so Phase 1 results survive a connection drop
        save_checkpoint(enriched, {}, {}, True, False)
        st.success(f"✅ Phase 1 done — {len(enriched)} responses generated.")

    if st.session_state.phase1_done and st.session_state.enriched:
        st.markdown("#### RAG Responses")
        for e in st.session_state.enriched:
            with st.expander(f"**{e['id']}** — {e['user_input']}"):
                st.markdown("**Retrieved contexts**")
                for j, ctx in enumerate(e["retrieved_contexts"]):
                    st.caption(f"Chunk {j+1}: {ctx[:200]}…")
                st.markdown("**RAG response**")
                st.success(e["response"])
                st.markdown("**Reference answer**")
                st.info(e["reference"])

    st.divider()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    st.markdown("### Phase 2 — RAGAS Metric Scoring")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Judge model", "llama-3.1-8b-instant")
    col_info2.metric("Sample cooldown", f"{SAMPLE_COOLDOWN}s")
    col_info3.metric("Experiment cooldown", f"{EXPERIMENT_COOLDOWN}s")

    st.write(
        "Scores each golden with 5 RAGAS metrics **one sample at a time** "
        "to stay within Groq's 6,000 TPM limit. "
        "Progress is **saved to disk after every experiment** — "
        "if the connection drops, use *Restore from checkpoint* in the sidebar."
    )

    if not st.session_state.phase1_done:
        st.warning("⚠️  Run Phase 1 first to generate RAG responses.")
    else:
        already_done = list(st.session_state.scores.keys())
        remaining = [n for n, _, _ in EXPERIMENTS if n not in already_done]

        if already_done:
            st.info(
                f"Experiments already completed: **{', '.join(already_done)}**\n\n"
                f"Remaining: **{', '.join(remaining) if remaining else 'none — all done!'}**"
            )

        btn_label = (
            "▶ Resume Phase 2" if already_done else "▶ Run Phase 2 — RAGAS Evaluation (~12-15 min)"
        )

        if remaining and st.button(btn_label, type="primary"):
            enriched = st.session_state.enriched
            judge_llm = build_judge(JUDGE_GROQ)
            ragas_emb = get_ragas_embeddings()

            total_experiments = len(EXPERIMENTS)
            overall_progress = st.progress(
                len(already_done) / total_experiments,
                text=f"Resuming from experiment {len(already_done)+1}…",
            )

            for exp_idx, (name, factory, keys) in enumerate(EXPERIMENTS):

                # Skip already-completed experiments (resume from checkpoint)
                if name in st.session_state.scores:
                    overall_progress.progress(
                        (exp_idx + 1) / total_experiments,
                        text=f"Skipping {name} (already done)",
                    )
                    continue

                st.markdown(
                    f"**[{exp_idx+1}/{total_experiments}] {name}** "
                    f"— {len(enriched)} samples, {SAMPLE_COOLDOWN}s between each"
                )

                metric = factory(judge_llm, ragas_emb)
                inputs = prepare_inputs(enriched, keys)

                # FIX 4: Collect errors per sample so they are visible in the UI
                exp_errors = []
                sample_lines = []
                sample_log = st.empty()

                def make_status_cb(log_ref, lines_ref):
                    def cb(i, total):
                        lines_ref.append(f"  [{i+1}/{total}] scoring…")
                        log_ref.code("\n".join(lines_ref[-10:]))
                    return cb

                def make_error_cb(errors_list, log_ref, lines_ref):
                    def cb(i, msg):
                        errors_list.append({"sample": i + 1, "error": msg})
                        lines_ref.append(f"  ⚠️  sample {i+1} error: {msg[:80]}")
                        log_ref.code("\n".join(lines_ref[-10:]))
                    return cb

                scores = _run(
                    score_experiment(
                        metric,
                        inputs,
                        status_callback=make_status_cb(sample_log, sample_lines),
                        error_callback=make_error_cb(exp_errors, sample_log, sample_lines),
                    )
                )

                # FIX 5: Persist scores to session state immediately after each experiment
                st.session_state.scores[name] = scores
                st.session_state.errors[name] = exp_errors

                # FIX 2: Save checkpoint to disk right away
                save_checkpoint(
                    enriched,
                    st.session_state.scores,
                    st.session_state.errors,
                    True,
                    False,
                )

                # Show result summary with error count
                avg = _avg(scores)
                null_count = sum(1 for s in scores if s is None)
                result_line = f"✅ **{name}**: {_badge(avg)} {avg:.2f}" if avg is not None else f"❌ **{name}**: all samples failed"
                if null_count:
                    result_line += f"  ⚠️ {null_count}/{len(scores)} sample(s) failed"
                sample_lines.append(result_line)
                sample_log.code("\n".join(sample_lines[-10:]))

                # Show per-sample errors as warnings
                if exp_errors:
                    with st.expander(f"⚠️ {len(exp_errors)} error(s) in {name}"):
                        for err in exp_errors:
                            st.warning(f"Sample {err['sample']}: {err['error']}")

                overall_progress.progress(
                    (exp_idx + 1) / total_experiments,
                    text=f"Completed {exp_idx+1}/{total_experiments} experiments",
                )

                # Countdown between experiments
                if exp_idx < total_experiments - 1:
                    next_name = EXPERIMENTS[exp_idx + 1][0]
                    if next_name not in st.session_state.scores:
                        countdown = st.empty()
                        for remaining_secs in range(EXPERIMENT_COOLDOWN, 0, -1):
                            countdown.info(
                                f"⏳ Cooldown before **{next_name}**: "
                                f"**{remaining_secs}s** remaining…"
                            )
                            time.sleep(1)
                        countdown.empty()

            # All experiments done
            overall_progress.progress(1.0, text="Phase 2 complete ✅")

            st.session_state.results = build_results(enriched, st.session_state.scores)
            st.session_state.phase2_done = True

            save_checkpoint(
                enriched,
                st.session_state.scores,
                st.session_state.errors,
                True,
                True,
            )
            save_results(RESULTS_PATH, st.session_state.results)

            st.success("✅ Phase 2 complete — results saved to results.json")
            st.balloons()

        elif not remaining:
            st.success("✅ All 5 experiments already completed. See the Results tab.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    st.subheader("Evaluation Results")

    # Try session state first, then checkpoint, then results.json
    if not st.session_state.phase2_done or not st.session_state.results:
        checkpoint = load_checkpoint()
        if checkpoint and checkpoint.get("phase2_done") and checkpoint.get("enriched"):
            st.session_state.results = build_results(
                checkpoint["enriched"], checkpoint.get("scores", {})
            )
            st.session_state.phase2_done = True
        elif os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, encoding="utf-8") as f:
                st.session_state.results = json.load(f)
            st.session_state.phase2_done = True
        else:
            # Show partial results if Phase 2 is in progress
            if st.session_state.scores:
                st.info(
                    f"Phase 2 in progress — showing partial results "
                    f"({len(st.session_state.scores)}/5 experiments done)."
                )
                partial_results = build_results(
                    st.session_state.enriched or [],
                    st.session_state.scores,
                )
                st.session_state.results = partial_results
            else:
                st.info("Run the full pipeline (Phase 1 + Phase 2) to see results here.")
                st.stop()

    results = st.session_state.results

    # ── Average metric cards ──────────────────────────────────────────────────
    st.markdown("#### Overall Averages")
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        avg = results["averages"].get(name)
        badge = _badge(avg)
        if avg is None:
            col.metric(label=name, value="⬜ N/A", help="Not yet scored")
        else:
            label = "✅ Good" if avg >= 0.75 else ("⚠️ Fair" if avg >= 0.5 else "❌ Poor")
            col.metric(label=name, value=f"{badge} {avg:.2f}", help=label)

    st.divider()

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown("#### Per-Golden Scores")
    rows = []
    for g in results["per_golden"]:
        row = {
            "ID": g["id"],
            "Metric Focus": g["metric_focus"],
            "Question": g["user_input"][:55] + "…",
        }
        for name in METRIC_NAMES:
            s = g["scores"].get(name)
            row[name[:12]] = f"{_badge(s)} {s:.2f}" if s is not None else "⬜ N/A"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # FIX 4: Show any errors that occurred during scoring
    all_errors = st.session_state.errors
    total_errors = sum(len(v) for v in all_errors.values())
    if total_errors:
        with st.expander(f"⚠️ {total_errors} scoring error(s) across all experiments"):
            for exp_name, errs in all_errors.items():
                if errs:
                    st.markdown(f"**{exp_name}**")
                    for err in errs:
                        st.warning(f"Sample {err['sample']}: {err['error']}")

    st.divider()

    # ── Per-golden detail ─────────────────────────────────────────────────────
    st.markdown("#### Per-Golden Detail")
    for g in results["per_golden"]:
        scores_str = "  |  ".join(
            f"{n[:8]}: {_badge(g['scores'].get(n))} {g['scores'].get(n):.2f}"
            if g["scores"].get(n) is not None
            else f"{n[:8]}: ⬜ N/A"
            for n in METRIC_NAMES
        )
        with st.expander(f"**{g['id']}** — {g['user_input']}"):
            st.caption(scores_str)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**RAG Response**")
                st.write(g["response"])
                st.markdown("**Reference**")
                st.info(g["reference"])
            with col_r:
                st.markdown("**Retrieved Contexts**")
                for j, ctx in enumerate(g.get("retrieved_contexts", [])):
                    st.caption(f"Chunk {j+1}")
                    st.write(ctx[:300] + ("…" if len(ctx) > 300 else ""))

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    st.download_button(
        label="⬇️  Download results.json",
        data=json.dumps(results, indent=2, ensure_ascii=False),
        file_name="technest_eval_results.json",
        mime="application/json",
    )
