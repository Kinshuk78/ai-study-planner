"""Generate an explanation grounded in retrieved chunks.

Hard invariant: every output must contain at least one ``[chunk_id]``
citation tag. The faithfulness evaluation depends on this.
"""

from __future__ import annotations

import re

from src.llm.provider import LLMProvider
from src.types import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[(?P<id>[A-Za-z0-9_\-]+)\]")


def explain(
    *,
    topic_name: str,
    question: str,
    mastery_level: float,
    chunks: list[RetrievedChunk],
    provider: LLMProvider,
) -> str:
    context = _format_chunks(chunks)
    text = provider.render_and_complete(
        "EXPLANATION",
        {
            "topic_name": topic_name,
            "question": question,
            "mastery_level": _bucket_mastery(mastery_level),
            "context": context,
        },
    )
    if not _CITATION_PATTERN.search(text):
        raise ValueError("explanation contains no [chunk_id] citation tags")
    return text


def extract_citations(text: str) -> list[str]:
    return [m.group("id") for m in _CITATION_PATTERN.finditer(text)]


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{rc.chunk.id}] {rc.chunk.text}" for rc in chunks)


def _bucket_mastery(p: float) -> str:
    if p < 0.3:
        return "novice"
    if p < 0.7:
        return "intermediate"
    return "advanced"
