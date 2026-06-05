from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from evals.metrics import METRIC_NAMES
from evals.reporter import _avg, _badge, build_results
from ui.checkpoint import RESULTS_PATH, load_checkpoint


def render_results_tab() -> None:
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

    # Show any errors that occurred during scoring
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
