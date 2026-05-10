"""Weekly summary via graph-traversal RAG."""

from __future__ import annotations

from src.llm.provider import LLMProvider
from src.types import GradedResponse, RetrievedChunk


def summarise_week(
    *,
    week_number: int,
    sessions_log: str,
    mastery_diff: dict[str, tuple[float, float]],
    chunks: list[RetrievedChunk],
    provider: LLMProvider,
) -> str:
    diff_lines = "\n".join(
        f"  {topic}: {before:.2f} -> {after:.2f}"
        for topic, (before, after) in sorted(mastery_diff.items())
    )
    context = "\n\n".join(f"[{rc.chunk.id}] {rc.chunk.text}" for rc in chunks)
    return provider.render_and_complete(
        "WEEKLY_SUMMARY",
        {
            "week_number": week_number,
            "sessions_log": sessions_log,
            "mastery_diff": diff_lines,
            "context": context,
        },
    )


def format_sessions_log(graded: list[GradedResponse]) -> str:
    lines = []
    for gr in graded:
        topic = gr.response.question.citations[0] if gr.response.question.citations else "?"
        lines.append(f"  {topic}: score {gr.score:.2f}")
    return "\n".join(lines) if lines else "  (no sessions)"
