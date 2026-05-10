"""Graph-traversal RAG — walks prerequisite edges and pools chunks.

This is the meaningful difference from flat top-k retrieval: when
explaining a topic or summarising a week, we deliberately surface
chunks from the queried topic's *prerequisites* so the LLM can ground
the explanation in concepts the learner has already seen.
"""

from __future__ import annotations

from src.config import load_config
from src.kg import KnowledgeGraph
from src.kg.traversal import walk_prerequisites
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.types import RetrievedChunk


def graph_traversal_retrieve(
    *,
    query: str,
    topic_id: str,
    kg: KnowledgeGraph,
    store: VectorStore,
    embedder: Embedder,
    max_depth: int | None = None,
    top_k_per_node: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve chunks for ``topic_id`` and its (transitive) prerequisites."""
    cfg = load_config()["rag"]["traversal"]
    max_depth = cfg["max_depth"] if max_depth is None else max_depth
    top_k_per_node = cfg["top_k_per_node"] if top_k_per_node is None else top_k_per_node

    nodes = walk_prerequisites(kg, topic_id, max_depth=max_depth)
    embedding = embedder.embed_text(query)

    pooled: dict[str, RetrievedChunk] = {}
    for node in nodes:
        for rc in store.query(embedding, top_k=top_k_per_node, topic_ids=[node]):
            existing = pooled.get(rc.chunk.id)
            if existing is None or rc.score > existing.score:
                pooled[rc.chunk.id] = rc

    return sorted(pooled.values(), key=lambda r: r.score, reverse=True)
