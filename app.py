"""TechNest RAG Evaluation Pipeline — streamlit run app.py"""
import streamlit as st

from ui.state import init_session_state
from ui.sidebar import render_sidebar
from ui.tabs.catalog import render_catalog_tab
from ui.tabs.goldens import render_goldens_tab
from ui.tabs.pipeline import render_pipeline_tab
from ui.tabs.results import render_results_tab

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

init_session_state()
render_sidebar()

tab_catalog, tab_goldens, tab_pipeline, tab_results = st.tabs(
    ["📚 Catalog", "🎯 Goldens", "🚀 Run Evaluation", "📊 Results"]
)

with tab_catalog:
    render_catalog_tab()

with tab_goldens:
    render_goldens_tab()

with tab_pipeline:
    render_pipeline_tab()

with tab_results:
    render_results_tab()
