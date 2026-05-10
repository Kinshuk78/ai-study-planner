"""Embedder facade.

Delegates to an :class:`LLMProvider`'s ``embed`` method by default. This
keeps the test path provider-agnostic — the :class:`MockProvider` ships
a deterministic hash-based embedding, so RAG tests run without
``sentence-transformers``.
"""

from __future__ import annotations

from src.llm.provider import LLMProvider
from src.types import Chunk


class Embedder:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def embed_text(self, text: str) -> list[float]:
        return self._provider.embed(text)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return [self.embed_text(c.text) for c in chunks]
