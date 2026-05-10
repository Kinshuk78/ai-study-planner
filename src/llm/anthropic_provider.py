"""Anthropic backend.

This is the **only** module outside the abstract layer that imports
``anthropic``.
"""

from __future__ import annotations

import os
from typing import Any

from src.config import load_config
from src.llm.provider import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("anthropic SDK is not installed; pip install anthropic") from exc

        cfg = load_config()["llm"]
        self._model = cfg["anthropic_model"]
        self._temperature = cfg["temperature"]
        self._max_tokens = cfg["max_tokens"]
        self._timeout = cfg["request_timeout_seconds"]
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=self._timeout,
        )
        self._embedder: Any | None = None  # lazily loaded sentence-transformers model

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # The Messages API returns a list of content blocks.
        parts = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    def embed(self, text: str) -> list[float]:
        # Anthropic does not currently expose an embedding API; we use
        # sentence-transformers locally so the provider is self-contained.
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            cfg = load_config()["rag"]
            self._embedder = SentenceTransformer(cfg["embedding_model"])
        vec = self._embedder.encode(text, convert_to_numpy=True)
        return list(vec.tolist())
