"""Vector store interface plus an in-memory cosine implementation.

The in-memory store is the default for tests and small demos. A
ChromaDB-backed implementation can be plugged in by subclassing
:class:`VectorStore`; both expose the same query API so callers don't
care which is in use.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from src.types import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        *,
        top_k: int,
        topic_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return ``top_k`` chunks most similar to ``embedding``.

        If ``topic_ids`` is given, restrict results to chunks whose
        ``topic_id`` is in the list.
        """

    @abstractmethod
    def __len__(self) -> int: ...


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int,
        topic_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        scored: list[tuple[float, Chunk]] = []
        topic_filter = set(topic_ids) if topic_ids is not None else None
        for chunk, vec in zip(self._chunks, self._embeddings, strict=True):
            if topic_filter is not None and (chunk.topic_id not in topic_filter):
                continue
            scored.append((_cosine(embedding, vec), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievedChunk(chunk=c, score=s) for s, c in scored[:top_k]]

    def all(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector dim mismatch")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
