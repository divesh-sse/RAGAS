from __future__ import annotations

import asyncio
from rag.retriever import Retriever
from rag.generator import Generator


async def run_phase1(
    goldens: list[dict],
    retriever: Retriever,
    generator: Generator,
    top_k: int = 3,
    spacing_s: int = 5,
) -> list[dict]:
    """
    Phase 1: run the RAG pipeline on every golden.
    Returns enriched list with retrieved_contexts and response added.
    No Streamlit calls — safe to run inside a ThreadPoolExecutor worker thread.
    """
    enriched = []
    for i, golden in enumerate(goldens):
        contexts = retriever.retrieve(golden["user_input"], top_k=top_k)
        response = await generator.generate(golden["user_input"], contexts)
        enriched.append(
            {
                **golden,
                "retrieved_contexts": contexts,
                "response": response,
            }
        )
        if i < len(goldens) - 1:
            await asyncio.sleep(spacing_s)
    return enriched
