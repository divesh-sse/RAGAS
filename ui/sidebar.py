from __future__ import annotations

import os

import streamlit as st

from evals.reporter import build_results
from ui.checkpoint import CHECKPOINT_PATH, load_checkpoint


def render_sidebar() -> None:
    with st.sidebar:

        # ── API Keys (Bring Your Own Keys) ────────────────────────────────────
        st.header("🔑 API Keys")
        st.caption("Keys are stored only in your browser session — never saved to disk.")

        groq_input = st.text_input(
            "GROQ_API_KEY",
            value=st.session_state.groq_key,
            type="password",
            placeholder="gsk_…",
            help="Used for RAG generation (llama-3.3-70b-versatile). Free key at console.groq.com.",
        )
        judge_input = st.text_input(
            "JUDGE_GROQ *(optional)*",
            value=st.session_state.judge_key,
            type="password",
            placeholder="gsk_… (falls back to GROQ_API_KEY)",
            help="Used for RAGAS evaluation (llama-3.1-8b-instant). Keeping it separate means eval runs never exhaust your main key.",
        )
        gemini_input = st.text_input(
            "GEMINI_API_KEY",
            value=st.session_state.gemini_key,
            type="password",
            placeholder="AIza…",
            help="Used for Gemini retrieval embeddings (gemini-embedding-2-preview). Free key at aistudio.google.com.",
        )

        st.session_state.groq_key   = groq_input
        st.session_state.judge_key  = judge_input or groq_input  # fallback if blank
        st.session_state.gemini_key = gemini_input

        keys_ready = st.session_state.groq_key and st.session_state.gemini_key
        if keys_ready:
            st.success("✅ Keys loaded")
        else:
            missing = []
            if not st.session_state.groq_key:
                missing.append("GROQ_API_KEY")
            if not st.session_state.gemini_key:
                missing.append("GEMINI_API_KEY")
            st.warning(f"Missing: {', '.join(missing)}")

        st.divider()

        # ── Session / Checkpoint ──────────────────────────────────────────────
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
                st.session_state.enriched    = checkpoint.get("enriched")
                st.session_state.scores      = checkpoint.get("scores", {})
                st.session_state.errors      = checkpoint.get("errors", {})
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
            for k in ["enriched", "scores", "errors", "results", "phase1_done", "phase2_done"]:
                st.session_state[k] = {} if k in ("scores", "errors") else (None if k not in ("phase1_done", "phase2_done") else False)
            st.rerun()
