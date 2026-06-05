from __future__ import annotations

import streamlit as st

from rag.loader import load_goldens


def render_goldens_tab() -> None:
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
