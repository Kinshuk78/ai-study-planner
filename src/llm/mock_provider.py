"""Mock LLM provider for deterministic, free unit/integration tests.

Returns deterministic responses keyed on the rendered prompt's
**system signature** (the first 64 chars of the system prompt). Each
task module ships a default response so tests work without
boilerplate; tests can override responses by mutating
:attr:`MockProvider.responses` directly.
"""

from __future__ import annotations

import hashlib
import json
import re
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
        dynamic = self._dynamic_response(system, user)
        if dynamic is not None:
            return dynamic
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

    def _dynamic_response(self, system: str, user: str) -> str | None:
        if self._matches_prompt_name(system, "QUIZ_GENERATION"):
            return self._quiz_response(user)
        if self._matches_prompt_name(system, "DISRUPTION_PARSING"):
            return self._disruption_response(user)
        if self._matches_prompt_name(system, "EXPLANATION"):
            return self._explanation_response(user)
        if self._matches_prompt_name(system, "WEEKLY_SUMMARY"):
            return self._weekly_summary_response(user)
        return None

    def _quiz_response(self, user: str) -> str:
        topic = _extract_value(user, "Topic") or "the topic"
        citations = _extract_chunk_ids(user)
        cite = citations[0] if citations else "chunk_demo_1"
        if "regression" in topic.lower():
            stem = "What does ordinary least squares minimise?"
            choices = ["Squared residuals", "Topic count", "Matrix size", "Attendance"]
            answer = "Squared residuals"
            short = "Why should regression assumptions be checked?"
            short_answer = "They affect whether model conclusions are reliable."
        else:
            stem = "Which structure can represent a dataset with rows and columns?"
            choices = ["Matrix", "Scalar", "Residual", "Deadline"]
            answer = "Matrix"
            short = "Why are projections useful for understanding regression?"
            short_answer = "They connect fitted values to geometric representations of data."
        return json.dumps(
            {
                "questions": [
                    {
                        "type": "mcq",
                        "stem": stem,
                        "choices": choices,
                        "answer": answer,
                        "citations": [cite],
                    },
                    {
                        "type": "short",
                        "stem": short,
                        "answer": short_answer,
                        "citations": [cite],
                    },
                ]
            }
        )

    def _disruption_response(self, user: str) -> str:
        today = _extract_value(user, "Today's date") or "2026-05-15"
        report = (_extract_value(user, "Self-report") or user).lower()
        if "minute" in report or "hour" in report or "capacity" in report:
            match = re.search(r"(\d+)\s*(?:minutes?|mins?)", report)
            minutes = int(match.group(1)) if match else 60
            payload = {"type": "capacity_change", "new_daily_minutes": minutes}
        elif "deadline" in report or "due" in report:
            match = re.search(r"\d{4}-\d{2}-\d{2}", user)
            payload = {"type": "deadline_change", "new_deadline": match.group(0) if match else today}
        elif "completed" in report or "finished" in report:
            payload = {"type": "completed_externally", "topic_ids": ["linear_algebra"]}
        else:
            payload = {"type": "sick_day", "date": today}
        payload["confidence"] = 0.95
        return json.dumps(payload)

    def _explanation_response(self, user: str) -> str:
        topic = _extract_value(user, "Topic") or "This topic"
        citations = _extract_chunk_ids(user)
        cite = citations[0] if citations else "chunk_demo_1"
        if "regression" in topic.lower():
            return (
                f"{topic} estimates how an outcome changes with predictors [{cite}]. "
                f"Ordinary least squares chooses coefficients by reducing squared residuals [{cite}]."
            )
        return (
            f"{topic} represents academic datasets using vectors and matrices [{cite}]. "
            f"These structures support later modelling steps such as projections and regression [{cite}]."
        )

    def _weekly_summary_response(self, user: str) -> str:
        citations = _extract_chunk_ids(user)
        cite = citations[0] if citations else "chunk_demo_1"
        focus = (_extract_value(user, "Focus topic") or "").lower()
        context = user.split("CONTEXT", 1)[-1].lower()
        if "regression" in focus or (not focus and "regression" in context):
            return (
                f"This week focused on regression modelling and residual-based fitting [{cite}]. "
                "Next, review assumptions before interpreting research claims."
            )
        return (
            f"This week focused on linear algebra foundations for representing datasets [{cite}]. "
            "Next, connect vectors, matrices, and projections to regression modelling."
        )


def _extract_value(text: str, label: str) -> str | None:
    pattern = rf"^{re.escape(label)}:\s*(.+)$"
    for line in text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1).strip().strip('"')
    return None


def _extract_chunk_ids(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_\-]+)\]", text)))
