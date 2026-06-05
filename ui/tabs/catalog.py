from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.cache import get_catalog


def render_catalog_tab() -> None:
    st.subheader("Knowledge Base — TechNest Product Catalog")
    st.write(
        "15 entries across **products**, **policies**, and **FAQs** "
        "that the RAG system retrieves from."
    )

    catalog = get_catalog()
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
