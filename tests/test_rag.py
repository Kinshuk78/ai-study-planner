from __future__ import annotations

import uuid

import pytest

from src.kg import KnowledgeGraph, Topic
from src.llm.mock_provider import MockProvider
from src.rag import (
    Embedder,
    InMemoryVectorStore,
    chunk_text,
    focused_retrieve,
    graph_traversal_retrieve,
)
from src.types import Chunk

# ---------- chunker ------------------------------------------------


def test_chunk_text_produces_uuid_ids():
    chunks = chunk_text("word " * 1200, source="x.txt", chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        uuid.UUID(c.id)  # must parse
        assert c.source == "x.txt"


def test_chunk_text_carries_topic_id():
    [c] = chunk_text(
        "short body", source="x.txt", topic_id="regression", chunk_size=500, chunk_overlap=50
    )
    assert c.topic_id == "regression"


def test_chunk_text_empty_input():
    assert chunk_text("", source="x.txt") == []


def test_chunk_text_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", source="x.txt", chunk_size=10, chunk_overlap=10)


def test_chunk_text_unique_ids_within_doc():
    chunks = chunk_text("word " * 1200, source="x.txt", chunk_size=300, chunk_overlap=50)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# ---------- vector store -------------------------------------------


def _store_with(chunks: list[Chunk], embedder: Embedder) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(chunks, embedder.embed_chunks(chunks))
    return store


def test_vectorstore_query_returns_top_k():
    embedder = Embedder(MockProvider())
    chunks = [
        Chunk(id="a", text="apples", source="x", topic_id="fruit"),
        Chunk(id="b", text="bananas", source="x", topic_id="fruit"),
        Chunk(id="c", text="zebras", source="x", topic_id="animal"),
    ]
    store = _store_with(chunks, embedder)
    results = store.query(embedder.embed_text("apples"), top_k=2)
    assert len(results) == 2


def test_vectorstore_topic_filter_excludes_other_topics():
    embedder = Embedder(MockProvider())
    chunks = [
        Chunk(id="a", text="apples", source="x", topic_id="fruit"),
        Chunk(id="b", text="zebras", source="x", topic_id="animal"),
    ]
    store = _store_with(chunks, embedder)
    results = store.query(embedder.embed_text("apples"), top_k=5, topic_ids=["fruit"])
    assert {r.chunk.id for r in results} == {"a"}


def test_vectorstore_length():
    embedder = Embedder(MockProvider())
    chunks = [Chunk(id=f"c{i}", text=f"text {i}", source="x", topic_id="t") for i in range(3)]
    store = _store_with(chunks, embedder)
    assert len(store) == 3


# ---------- focused RAG --------------------------------------------


def test_focused_retrieve_only_returns_topic_chunks():
    embedder = Embedder(MockProvider())
    chunks = [
        Chunk(id="a", text="regression analysis is", source="x", topic_id="regression"),
        Chunk(id="b", text="zebra patterns evolved", source="x", topic_id="animal"),
    ]
    store = _store_with(chunks, embedder)
    results = focused_retrieve(
        query="regression", topic_id="regression", store=store, embedder=embedder, top_k=5
    )
    assert all(r.chunk.topic_id == "regression" for r in results)
    assert {r.chunk.id for r in results} == {"a"}


# ---------- graph-traversal RAG ------------------------------------


def test_traversal_includes_prerequisite_chunks():
    embedder = Embedder(MockProvider())
    kg = KnowledgeGraph()
    for tid in ("linear_algebra", "regression"):
        kg.add_topic(Topic(id=tid, name=tid))
    kg.add_prerequisite("linear_algebra", "regression")

    chunks = [
        Chunk(id="la", text="vectors and matrices", source="x", topic_id="linear_algebra"),
        Chunk(id="reg", text="line of best fit", source="x", topic_id="regression"),
        Chunk(id="other", text="zebra", source="x", topic_id="animal"),
    ]
    store = _store_with(chunks, embedder)

    results = graph_traversal_retrieve(
        query="regression",
        topic_id="regression",
        kg=kg,
        store=store,
        embedder=embedder,
        max_depth=2,
        top_k_per_node=2,
    )
    ids = {r.chunk.id for r in results}
    assert "reg" in ids
    assert "la" in ids
    assert "other" not in ids


def test_traversal_respects_max_depth():
    embedder = Embedder(MockProvider())
    kg = KnowledgeGraph()
    for tid in ("a", "b", "c"):
        kg.add_topic(Topic(id=tid, name=tid))
    kg.add_prerequisite("a", "b")
    kg.add_prerequisite("b", "c")

    chunks = [
        Chunk(id=tid, text=f"about {tid}", source="x", topic_id=tid) for tid in ("a", "b", "c")
    ]
    store = _store_with(chunks, embedder)

    results = graph_traversal_retrieve(
        query="c",
        topic_id="c",
        kg=kg,
        store=store,
        embedder=embedder,
        max_depth=1,
        top_k_per_node=2,
    )
    ids = {r.chunk.id for r in results}
    assert "c" in ids
    assert "b" in ids
    assert "a" not in ids  # depth-2 ancestor pruned
