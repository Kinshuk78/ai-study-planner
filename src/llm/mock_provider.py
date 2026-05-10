"""Mock LLM provider for deterministic, free unit/integration tests.

Returns canned responses keyed on the rendered prompt's
**system signature** (the first 64 chars of the system prompt). Each
task module ships a default canned response so tests work without
boilerplate; tests can override responses by mutating
:attr:`MockProvider.responses` directly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.llm.provider import LLMProvider

DEFAULT_RESPONSES: dict[str, str] = {
    # Keyed by prompt name (caller is expected to set responses keyed
    # by the *full* signature when overriding).
    "KG_EXTRACTION": json.dumps(
        {
            "topics": [
                {
                    "id": "linear_algebra",
                    "name": "Linear Algebra",
                    "description": "vectors, matrices",
                },
                {"id": "regression", "name": "Regression", "description": "linear models"},
            ],
            "edges": [{"source": "linear_algebra", "target": "regression"}],
        }
    ),
    "QUIZ_GENERATION": json.dumps(
        {
            "questions": [
                {
                    "type": "mcq",
                    "stem": "What is 2 + 2?",
                    "choices": ["3", "4", "5", "6"],
                    "answer": "4",
                    "citations": ["chunk_demo_1"],
                },
                {
                    "type": "short",
                    "stem": "Define linearity.",
                    "answer": "additivity and homogeneity",
                    "citations": ["chunk_demo_2"],
                },
            ]
        }
    ),
    "GRADING_FREE_RESPONSE": json.dumps(
        {
            "score": 1.0,
            "feedback": "Correct.",
        }
    ),
    "DISRUPTION_PARSING": json.dumps(
        {
            "type": "sick_day",
            "date": "2026-05-09",
            "confidence": 0.95,
        }
    ),
    "EXPLANATION": (
        "Linear regression fits a line by minimising squared residuals "
        "[chunk_demo_1]. It depends on linear algebra concepts "
        "[chunk_demo_2]."
    ),
    "WEEKLY_SUMMARY": (
        "This week you covered linear algebra basics [chunk_demo_1]. "
        "Mastery is steady; next week focus on regression."
    ),
}


class MockProvider(LLMProvider):
    def __init__(self) -> None:
        self.responses: dict[str, str] = {}
        self.calls: list[dict[str, str]] = []

    # ----- core API ------------------------------------------------

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        self.calls.append({"system": system, "user": user})

        # Exact-signature override.
        sig = self._signature(system, user)
        if sig in self.responses:
            return self.responses[sig]

        # Fall back to prompt-name lookup based on the system prompt.
        for name, default in DEFAULT_RESPONSES.items():
            if name in self.responses:
                continue
            if self._matches_prompt_name(system, name):
                return default
        for name, default in DEFAULT_RESPONSES.items():
            if self._matches_prompt_name(system, name):
                return default

        raise KeyError(
            f"MockProvider has no response for signature '{sig}'. "
            f"Set provider.responses[<sig>] before the call, or extend DEFAULT_RESPONSES."
        )

    def embed(self, text: str) -> list[float]:
        # Deterministic 8-d hash-based embedding (no model dependency).
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[:8]]

    # ----- helpers -------------------------------------------------

    def set_response(self, system: str, user: str, response: str) -> None:
        """Register an exact (system, user) → response override."""
        self.responses[self._signature(system, user)] = response

    @staticmethod
    def _signature(system: str, user: str) -> str:
        return hashlib.sha256((system + "|" + user).encode("utf-8")).hexdigest()

    @staticmethod
    def _matches_prompt_name(system: str, name: str) -> bool:
        # Match by detecting prompt-specific keywords in the system text.
        markers = {
            "KG_EXTRACTION": "curriculum designer",
            "QUIZ_GENERATION": "formative quiz",
            "GRADING_FREE_RESPONSE": "impartial grader",
            "DISRUPTION_PARSING": "free-text learner self-report",
            "EXPLANATION": "explain concept",
            "WEEKLY_SUMMARY": "weekly summar",
        }
        return markers[name].lower() in system.lower()
