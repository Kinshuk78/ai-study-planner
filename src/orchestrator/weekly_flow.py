"""Weekly flow.

ingest week's materials (chunk + embed + link to KG nodes) -> advance
scheduler -> replan -> LLM weekly summary via graph-traversal RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.bkt import BKTPredictor
from src.kg import KnowledgeGraph
from src.llm.provider import LLMProvider
from src.llm.tasks.summariser import summarise_week
from src.rag import (
    Embedder,
    chunk_pdf,
    chunk_text,
    graph_traversal_retrieve,
)
from src.rag.vectorstore import VectorStore


@dataclass
class WeeklyResult:
    summary: str
    chunks_added: int


def run_weekly(
    *,
    week_number: int,
    materials: list[tuple[str, str]],  # [(topic_id, text or path)]
    kg: KnowledgeGraph,
    predictor: BKTPredictor,
    store: VectorStore,
    provider: LLMProvider,
    mastery_diff: dict[str, tuple[float, float]],
    sessions_log: str,
    half_life_days: float = 7.0,
) -> WeeklyResult:
    embedder = Embedder(provider)

    chunks_added = 0
    for topic_id, body in materials:
        if not kg.has_topic(topic_id):
            raise KeyError(f"unknown topic '{topic_id}' in weekly materials")
        if _looks_like_pdf_path(body):
            new_chunks = chunk_pdf(body, topic_id=topic_id)
        else:
            new_chunks = chunk_text(body, source=f"week_{week_number}", topic_id=topic_id)
        if not new_chunks:
            continue
        store.add(new_chunks, embedder.embed_chunks(new_chunks))
        chunks_added += len(new_chunks)

    # Advance the scheduler — apply forgetting decay to all topics.
    predictor.apply_decay_all(days_elapsed=7.0, half_life_days=half_life_days)

    # Pick the most-recently-touched topic for graph-traversal context.
    if mastery_diff:
        focus_topic = max(mastery_diff.items(), key=lambda kv: abs(kv[1][1] - kv[1][0]))[0]
    else:
        focus_topic = next(iter(kg.topics())).id

    context_chunks = graph_traversal_retrieve(
        query=f"summary of {focus_topic}",
        topic_id=focus_topic,
        kg=kg,
        store=store,
        embedder=embedder,
    )

    summary = summarise_week(
        week_number=week_number,
        sessions_log=sessions_log,
        mastery_diff=mastery_diff,
        chunks=context_chunks,
        provider=provider,
    )
    return WeeklyResult(summary=summary, chunks_added=chunks_added)


def _looks_like_pdf_path(body: str) -> bool:
    """Heuristic: treat ``body`` as a path only when it's short and ends in .pdf."""
    if len(body) > 1024 or "\n" in body:
        return False
    if not body.lower().endswith(".pdf"):
        return False
    try:
        return Path(body).exists()
    except (OSError, ValueError):
        return False
