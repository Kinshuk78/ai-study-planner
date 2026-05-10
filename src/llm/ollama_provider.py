"""Local Ollama backend.

This is the **only** module outside the abstract layer that imports
``ollama``.
"""

from __future__ import annotations

from typing import Any

from src.config import load_config
from src.llm.provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ollama SDK is not installed; pip install ollama") from exc

        cfg = load_config()["llm"]
        self._model = cfg["ollama_model"]
        self._temperature = cfg["temperature"]
        self._client = ollama
        self._embedder: Any | None = None

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": kwargs.get("temperature", self._temperature),
                "num_predict": kwargs.get("max_tokens", -1),
            },
        )
        return str(response["message"]["content"])

    def embed(self, text: str) -> list[float]:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            cfg = load_config()["rag"]
            self._embedder = SentenceTransformer(cfg["embedding_model"])
        vec = self._embedder.encode(text, convert_to_numpy=True)
        return list(vec.tolist())
