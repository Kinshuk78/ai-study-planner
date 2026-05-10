"""Prerequisite-edge traversal helpers used by graph-traversal RAG."""

from __future__ import annotations

from collections import deque

from src.kg.store import KnowledgeGraph


def ancestors(kg: KnowledgeGraph, topic_id: str, max_depth: int | None = None) -> list[str]:
    """Topics that are (transitive) prerequisites of ``topic_id``.

    Returned in BFS order starting from ``topic_id``'s direct prerequisites.
    The queried topic is **not** included.
    """
    if not kg.has_topic(topic_id):
        raise KeyError(f"unknown topic '{topic_id}'")
    visited: set[str] = set()
    order: list[str] = []
    queue: deque[tuple[str, int]] = deque((p, 1) for p in kg.get_prerequisites(topic_id))
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        if max_depth is not None and depth > max_depth:
            continue
        visited.add(node)
        order.append(node)
        for parent in kg.get_prerequisites(node):
            if parent not in visited:
                queue.append((parent, depth + 1))
    return order


def walk_prerequisites(
    kg: KnowledgeGraph, topic_id: str, max_depth: int | None = None
) -> list[str]:
    """The queried topic plus its (transitive) prerequisites.

    This is the node set used by graph-traversal RAG when retrieving
    chunks for cross-topic generation (explanations, weekly summaries).
    """
    return [topic_id, *ancestors(kg, topic_id, max_depth=max_depth)]
