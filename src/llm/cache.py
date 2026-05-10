"""Cache decorator for :class:`LLMProvider`.

Wraps any provider and persists responses to a JSONL file so subsequent
calls with the same ``(system, user, kwargs)`` are served from disk —
the canonical example syllabus then runs without API calls (CI-safe)
and the demo runs offline.

Design choices:

* **Decorator pattern** — works with any ``LLMProvider`` subclass.
  Embeddings pass through unchanged (sentence-transformers is local
  and already fast).
* **Append-only JSONL** — each line is a self-contained record so the
  file is human-readable, easy to diff, and resilient to mid-write
  crashes.
* **Cache key** — ``sha256(system + "|" + user + "|" + json(kwargs))``.
  Including kwargs means changing temperature / max_tokens correctly
  invalidates the cache.
* **Mock provider is never cached** — tests should be deterministic
  without filesystem state, and the mock already returns canned
  responses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.llm.provider import LLMProvider


class CachedProvider(LLMProvider):
    """Wrap an :class:`LLMProvider` with a JSONL response cache."""

    def __init__(self, inner: LLMProvider, cache_path: str | Path) -> None:
        self.inner = inner
        self.cache_path = Path(cache_path)
        self._cache: dict[str, str] = self._load()
        self.hits = 0
        self.misses = 0

    # ----- LLMProvider interface ------------------------------------

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        key = self._signature(system, user, kwargs)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        response = self.inner.complete(system, user, **kwargs)
        self._cache[key] = response
        self._append(key, system, user, kwargs, response)
        return response

    def embed(self, text: str) -> list[float]:
        # Embeddings are produced locally by sentence-transformers —
        # caching them adds I/O without saving anything meaningful.
        return self.inner.embed(text)

    # ----- introspection -------------------------------------------

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._cache)}

    # ----- persistence ---------------------------------------------

    def _load(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        cache: dict[str, str] = {}
        with open(self.cache_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines rather than crashing — the
                    # cache is best-effort, never load-bearing.
                    continue
                if "key" in entry and "response" in entry:
                    cache[entry["key"]] = entry["response"]
        return cache

    def _append(
        self,
        key: str,
        system: str,
        user: str,
        kwargs: dict[str, Any],
        response: str,
    ) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "system": system,
            "user": user,
            "kwargs": _stringify_kwargs(kwargs),
            "response": response,
        }
        with open(self.cache_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    # ----- helpers --------------------------------------------------

    @staticmethod
    def _signature(system: str, user: str, kwargs: dict[str, Any]) -> str:
        payload = f"{system}|{user}|{json.dumps(_stringify_kwargs(kwargs), sort_keys=True)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stringify_kwargs(kwargs: dict[str, Any]) -> dict[str, str]:
    """Serialise kwargs to a stable string-only dict so the cache key is
    deterministic across runs."""
    return {k: str(v) for k, v in sorted(kwargs.items())}
