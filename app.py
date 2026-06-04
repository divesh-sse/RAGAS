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
nest_asyncio.apply()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JUDGE_GROQ = os.getenv("JUDGE_GROQ", GROQ_API_KEY)


def _run(coro):
    """Run an async coroutine from Streamlit's sync context."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ── Cached heavy objects ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading catalog and embeddings…")
def get_retriever():
    catalog = load_catalog()
    return Retriever(catalog), catalog


@st.cache_resource(show_spinner="Loading RAGAS embeddings…")
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
    "scores": None,
    "results": None,
    "phase1_done": False,
    "phase2_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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
    st.dataframe(filtered, use_container_width=True, hide_index=True)
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
        "retrieves context chunks, generates an answer."
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

        with st.spinner("Running RAG on 5 goldens (5 s spacing between calls)…"):
            enriched = _run(
                run_phase1(goldens, retriever, generator, status_callback=phase1_cb)
            )

        progress.progress(1.0, text="Phase 1 complete ✅")
        st.session_state.enriched = enriched
        st.session_state.phase1_done = True
        st.success(f"✅ Phase 1 done — {len(enriched)} responses generated.")

    # Show Phase 1 results if available
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
        "Scores each golden with 5 RAGAS metrics, **one sample at a time** "
        "to stay within the 6,000 TPM Groq rate limit."
    )

    if not st.session_state.phase1_done:
        st.warning("⚠️  Run Phase 1 first to generate RAG responses.")
    else:
        if st.button("▶ Run Phase 2 — RAGAS Evaluation (~10-15 min)", type="primary"):
            enriched = st.session_state.enriched
            judge_llm = build_judge(JUDGE_GROQ)
            ragas_emb = get_ragas_embeddings()

            all_scores = {}
            total_experiments = len(EXPERIMENTS)

            overall_progress = st.progress(0, text="Starting RAGAS evaluation…")
            exp_status = st.empty()
            sample_log = st.empty()

            for exp_idx, (name, factory, keys) in enumerate(EXPERIMENTS):
                metric = factory(judge_llm, ragas_emb)
                inputs = prepare_inputs(enriched, keys)

                exp_status.markdown(
                    f"**[{exp_idx+1}/{total_experiments}] {name}** "
                    f"— {len(inputs)} samples, {SAMPLE_COOLDOWN}s between each"
                )

                sample_lines = []

                def make_sample_cb(exp_name, log_ref, lines_ref):
                    def cb(i, total):
                        lines_ref.append(f"  sample {i+1}/{total} scoring…")
                        log_ref.code("\n".join(lines_ref[-8:]))
                    return cb

                scores = _run(
                    score_experiment(
                        metric,
                        inputs,
                        status_callback=make_sample_cb(name, sample_log, sample_lines),
                    )
                )

                all_scores[name] = scores
                avg = _avg(scores)
                badge = _badge(avg)
                sample_lines.append(
                    f"  ✅ {name} done — avg {badge} {avg:.2f}"
                )
                sample_log.code("\n".join(sample_lines[-8:]))

                overall_progress.progress(
                    (exp_idx + 1) / total_experiments,
                    text=f"Completed {exp_idx+1}/{total_experiments} experiments",
                )

                # Cooldown countdown between experiments
                if exp_idx < total_experiments - 1:
                    countdown = st.empty()
                    for remaining in range(EXPERIMENT_COOLDOWN, 0, -1):
                        countdown.info(
                            f"⏳ Cooldown between experiments: **{remaining}s** remaining…"
                        )
                        time.sleep(1)
                    countdown.empty()

            overall_progress.progress(1.0, text="Phase 2 complete ✅")
            exp_status.empty()

            st.session_state.scores = all_scores
            st.session_state.results = build_results(enriched, all_scores)
            st.session_state.phase2_done = True

            save_results("results.json", st.session_state.results)
            st.success("✅ Phase 2 complete — results saved to results.json")
            st.balloons()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    st.subheader("Evaluation Results")

    if not st.session_state.phase2_done or not st.session_state.results:
        # Try loading from file if exists
        if os.path.exists("results.json"):
            with open("results.json", encoding="utf-8") as f:
                st.session_state.results = json.load(f)
            st.session_state.phase2_done = True
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
        label = "✅ Good" if avg and avg >= 0.75 else ("⚠️ Fair" if avg and avg >= 0.5 else "❌ Poor")
        col.metric(
            label=name,
            value=f"{badge} {avg:.2f}" if avg is not None else "N/A",
            help=label,
        )

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

    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

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
                for j, ctx in enumerate(g["retrieved_contexts"]):
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
