from __future__ import annotations

import streamlit as st

from evals.metrics import build_embeddings
from rag.loader import load_catalog
from rag.retriever import Retriever


@st.cache_resource(show_spinner="Loading catalog…")
def get_catalog():
    return load_catalog()


@st.cache_resource(show_spinner="Embedding catalog via Gemini API…")
def get_retriever(gemini_key: str):
    catalog = get_catalog()
    return Retriever(catalog, api_key=gemini_key)


@st.cache_resource(show_spinner="Loading RAGAS embeddings (sentence-transformers)…")
def get_ragas_embeddings():
    return build_embeddings()
