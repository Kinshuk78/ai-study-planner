"""Focused RAG — KG-scoped similarity search.

Used by single-topic generation (quizzes, focused explanations). Only
chunks whose ``topic_id`` matches the queried topic are eligible.
"""

from __future__ import annotations

from src.config import load_config
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.types import RetrievedChunk


def focused_retrieve(
    *,
    query: str,
    topic_id: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    cfg = load_config()["rag"]["focused"]
    top_k = top_k or cfg["top_k"]
    embedding = embedder.embed_text(query)
    return store.query(embedding, top_k=top_k, topic_ids=[topic_id])
