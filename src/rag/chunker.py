"""Text and PDF chunking with metadata.

Chunk IDs are UUIDs (hard invariant). Chunks carry ``source``, ``page``,
and optional ``topic_id`` so the vector store can do metadata filtering.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from src.config import load_config
from src.types import Chunk


def chunk_text(
    text: str,
    *,
    source: str,
    topic_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a string into overlapping word-window chunks."""
    cfg = load_config()["rag"]
    chunk_size = chunk_size or cfg["chunk_size"]
    chunk_overlap = chunk_overlap or cfg["chunk_overlap"]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        body = " ".join(window)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=body,
                source=source,
                page=None,
                topic_id=topic_id,
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_pdf(
    path: str | Path,
    *,
    topic_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Parse a PDF and chunk per-page so each chunk carries a ``page`` number."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pypdf is not installed; pip install pypdf") from exc

    path = Path(path)
    reader = PdfReader(str(path))
    out: list[Chunk] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_chunks = chunk_text(
            text,
            source=path.name,
            topic_id=topic_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for c in page_chunks:
            out.append(
                Chunk(id=c.id, text=c.text, source=c.source, page=page_index, topic_id=c.topic_id)
            )
    return out
