from __future__ import annotations

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class Retriever:
    def __init__(self, catalog: list[dict], api_key: str):
        self.catalog = catalog
        self._embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=api_key,
        )
        self._embeddings = self._embed_catalog()

    def _embed_catalog(self) -> np.ndarray:
        texts = [f"{item['title']}. {item['content']}" for item in self.catalog]
        vecs = self._embedder.embed_documents(texts)
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def _embed_query(self, query: str) -> np.ndarray:
        vec = np.array(self._embedder.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        qvec = self._embed_query(query)
        scores = self._embeddings @ qvec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.catalog[i]["content"] for i in top_indices]

    def retrieve_with_titles(self, query: str, top_k: int = 3) -> list[dict]:
        qvec = self._embed_query(query)
        scores = self._embeddings @ qvec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "title": self.catalog[i]["title"],
                "content": self.catalog[i]["content"],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]
