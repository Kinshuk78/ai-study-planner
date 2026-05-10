"""Tests for :class:`CachedProvider`.

Strategy: wrap a tiny counting provider so we can assert how many real
calls reach the inner provider. Hits should not increment the inner
counter; misses should and should also persist to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.llm.cache import CachedProvider
from src.llm.provider import LLMProvider


class CountingProvider(LLMProvider):
    """LLMProvider that counts calls and returns a deterministic
    response derived from the user prompt."""

    def __init__(self) -> None:
        self.complete_calls = 0
        self.embed_calls = 0

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.complete_calls += 1
        return f"reply to '{user}' (call #{self.complete_calls})"

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [float(len(text))]


def test_cache_misses_pass_through_to_inner(tmp_path: Path) -> None:
    inner = CountingProvider()
    cache = CachedProvider(inner, tmp_path / "cache.jsonl")

    result = cache.complete("sys", "user-a")
    assert result == "reply to 'user-a' (call #1)"
    assert inner.complete_calls == 1
    assert cache.stats() == {"hits": 0, "misses": 1, "size": 1}


def test_cache_hits_skip_inner(tmp_path: Path) -> None:
    inner = CountingProvider()
    cache = CachedProvider(inner, tmp_path / "cache.jsonl")

    first = cache.complete("sys", "user-a")
    second = cache.complete("sys", "user-a")

    assert first == second
    assert inner.complete_calls == 1  # only the first call hit the inner provider
    assert cache.stats() == {"hits": 1, "misses": 1, "size": 1}


def test_cache_distinguishes_different_prompts(tmp_path: Path) -> None:
    inner = CountingProvider()
    cache = CachedProvider(inner, tmp_path / "cache.jsonl")

    a = cache.complete("sys", "user-a")
    b = cache.complete("sys", "user-b")

    assert a != b
    assert inner.complete_calls == 2
    assert cache.stats()["size"] == 2


def test_cache_kwargs_change_invalidates_entry(tmp_path: Path) -> None:
    """Different temperature must hit a different cache entry."""
    inner = CountingProvider()
    cache = CachedProvider(inner, tmp_path / "cache.jsonl")

    cache.complete("sys", "u", temperature=0.0)
    cache.complete("sys", "u", temperature=1.0)
    cache.complete("sys", "u", temperature=0.0)  # hit

    assert inner.complete_calls == 2
    assert cache.stats() == {"hits": 1, "misses": 2, "size": 2}


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    inner_a = CountingProvider()
    cache_a = CachedProvider(inner_a, cache_path)
    cache_a.complete("sys", "user-a")
    cache_a.complete("sys", "user-b")

    # Now construct a fresh CachedProvider — it should pick up the
    # JSONL and serve both entries from cache without calling the inner.
    inner_b = CountingProvider()
    cache_b = CachedProvider(inner_b, cache_path)
    assert len(cache_b) == 2

    cache_b.complete("sys", "user-a")
    cache_b.complete("sys", "user-b")
    assert inner_b.complete_calls == 0
    assert cache_b.stats() == {"hits": 2, "misses": 0, "size": 2}


def test_cache_jsonl_records_are_self_describing(tmp_path: Path) -> None:
    cache = CachedProvider(CountingProvider(), tmp_path / "cache.jsonl")
    cache.complete("system prompt", "user prompt", temperature=0.7)

    lines = (tmp_path / "cache.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["system"] == "system prompt"
    assert record["user"] == "user prompt"
    assert record["kwargs"] == {"temperature": "0.7"}
    assert "reply to 'user prompt'" in record["response"]
    assert "key" in record


def test_cache_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text("not json\n" + json.dumps({"key": "k1", "response": "r1"}) + "\n")
    cache = CachedProvider(CountingProvider(), path)
    assert len(cache) == 1


def test_cache_does_not_cache_embeddings(tmp_path: Path) -> None:
    inner = CountingProvider()
    cache = CachedProvider(inner, tmp_path / "cache.jsonl")
    cache.embed("hello")
    cache.embed("hello")
    assert inner.embed_calls == 2
