from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv


def init_session_state() -> None:
    load_dotenv()  # no-op on Streamlit Cloud; pre-fills keys locally from .env
    defaults = {
        # BYOK — pre-filled from .env if present locally, blank on Streamlit Cloud
        "groq_key":   os.getenv("GROQ_API_KEY", ""),
        "judge_key":  os.getenv("JUDGE_GROQ", ""),
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        # Pipeline state
        "enriched":    None,
        "scores":      {},
        "errors":      {},
        "results":     None,
        "phase1_done": False,
        "phase2_done": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
