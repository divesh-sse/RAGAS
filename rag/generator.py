from __future__ import annotations

from openai import AsyncOpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a helpful customer support assistant for TechNest, an online electronics store.
Answer the customer's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer fully, say so honestly.
Keep your answer concise, factual, and friendly. Do not invent any details not present in the context."""


class Generator:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self._client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self._model = model

    async def generate(self, query: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nCustomer question: {query}",
            },
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
