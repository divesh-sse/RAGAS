from __future__ import annotations

import time

import streamlit as st

from evals.metrics import (
    EXPERIMENT_COOLDOWN,
    EXPERIMENTS,
    SAMPLE_COOLDOWN,
    build_judge,
    prepare_inputs,
    score_experiment,
)
from evals.reporter import _avg, _badge, build_results, save_results
from evals.runner import run_phase1
from rag.generator import Generator
from rag.loader import load_goldens
from ui.async_utils import run as _run
from ui.cache import get_ragas_embeddings, get_retriever
from ui.checkpoint import RESULTS_PATH, save_checkpoint


def render_pipeline_tab() -> None:
    st.subheader("Evaluation Pipeline")

    if not st.session_state.groq_key or not st.session_state.gemini_key:
        st.error("⚠️  GROQ_API_KEY and GEMINI_API_KEY are both required — enter them in the sidebar.")
        st.stop()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    st.markdown("### Phase 1 — RAG Pipeline")
    st.write(
        "Runs the RAG system on each golden question: "
        "retrieves context chunks, then generates an answer with Groq."
    )

    if st.button("▶ Run Phase 1 — Generate RAG Responses", type="primary"):
        if not st.session_state.gemini_key:
            st.error("⚠️  GEMINI_API_KEY required for retrieval — enter it in the sidebar.")
            st.stop()
        retriever = get_retriever(st.session_state.gemini_key)
        generator = Generator(api_key=st.session_state.groq_key)
        goldens = load_goldens()

        # _run() executes in a worker thread — no st.* calls inside
        with st.spinner(f"Running RAG on {len(goldens)} goldens (5s spacing)…"):
            enriched = _run(run_phase1(goldens, retriever, generator))

        # Back in main thread — safe to call st.* here
        st.session_state.enriched = enriched
        st.session_state.phase1_done = True
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
            judge_llm = build_judge(st.session_state.judge_key)
            ragas_emb = get_ragas_embeddings()
            total_experiments = len(EXPERIMENTS)
            overall_progress = st.progress(
                len(already_done) / total_experiments,
                text=f"Resuming from experiment {len(already_done)+1}…",
            )

            for exp_idx, (name, factory, keys) in enumerate(EXPERIMENTS):

                if name in st.session_state.scores:
                    overall_progress.progress(
                        (exp_idx + 1) / total_experiments,
                        text=f"Skipping {name} (already done)",
                    )
                    continue

                # ── All st.* calls here are in the main thread ── ✅
                st.markdown(f"**[{exp_idx+1}/{total_experiments}] {name}**")

                metric = factory(judge_llm, ragas_emb)
                inputs = prepare_inputs(enriched, keys)
                exp_errors = []  # plain list — appended to inside the worker thread

                # _run() blocks until the experiment finishes — no st.* inside
                with st.spinner(f"Scoring {len(inputs)} samples for {name}…"):
                    scores = _run(
                        score_experiment(metric, inputs, error_collector=exp_errors)
                    )

                # ── Back in main thread — safe to update UI ── ✅
                st.session_state.scores[name] = scores
                st.session_state.errors[name] = exp_errors
                save_checkpoint(enriched, st.session_state.scores, st.session_state.errors, True, False)

                avg = _avg(scores)
                null_count = sum(1 for s in scores if s is None)
                if avg is not None:
                    msg = f"✅ **{name}**: {_badge(avg)} {avg:.2f}"
                    if null_count:
                        msg += f"  ⚠️ {null_count}/{len(scores)} failed"
                    st.success(msg)
                else:
                    st.error(f"❌ **{name}**: all samples failed to score")

                if exp_errors:
                    with st.expander(f"⚠️ {len(exp_errors)} error(s) in {name}"):
                        for err in exp_errors:
                            st.warning(f"Sample {err['sample']}: {err['error']}")

                overall_progress.progress(
                    (exp_idx + 1) / total_experiments,
                    text=f"Completed {exp_idx+1}/{total_experiments} experiments",
                )

                # Cooldown between experiments — time.sleep is fine in main thread
                if exp_idx < total_experiments - 1:
                    next_name = EXPERIMENTS[exp_idx + 1][0]
                    if next_name not in st.session_state.scores:
                        countdown = st.empty()
                        for remaining_secs in range(EXPERIMENT_COOLDOWN, 0, -1):
                            countdown.info(f"⏳ Cooldown before **{next_name}**: **{remaining_secs}s**…")
                            time.sleep(1)
                        countdown.empty()

            overall_progress.progress(1.0, text="Phase 2 complete ✅")
            st.session_state.results = build_results(enriched, st.session_state.scores)
            st.session_state.phase2_done = True
            save_checkpoint(enriched, st.session_state.scores, st.session_state.errors, True, True)
            save_results(RESULTS_PATH, st.session_state.results)
            st.success("✅ Phase 2 complete — results saved to results.json")
            st.balloons()

        elif not remaining:
            st.success("✅ All 5 experiments already completed. See the Results tab.")
