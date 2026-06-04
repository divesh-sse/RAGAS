import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, catalog: list[dict], model_name: str = MODEL_NAME):
        self.catalog = catalog
        self._model = SentenceTransformer(model_name)
        self._embeddings = self._embed_catalog()

    def _embed_catalog(self) -> np.ndarray:
        texts = [f"{item['title']}. {item['content']}" for item in self.catalog]
        return self._model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_emb = self._model.encode([query], normalize_embeddings=True)
        scores = (query_emb @ self._embeddings.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.catalog[i]["content"] for i in top_indices]

    def retrieve_with_titles(self, query: str, top_k: int = 3) -> list[dict]:
        query_emb = self._model.encode([query], normalize_embeddings=True)
        scores = (query_emb @ self._embeddings.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "title": self.catalog[i]["title"],
                "content": self.catalog[i]["content"],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]
