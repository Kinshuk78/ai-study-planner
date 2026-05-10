"""Integration tests for the four orchestrator flows.

All tests use :class:`MockProvider` so they're deterministic and free.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.config import load_config, seed_everything
from src.kg import KnowledgeGraph
from src.llm.mock_provider import MockProvider
from src.orchestrator import handle_disruption, run_session, run_setup, run_weekly
from src.rag import InMemoryVectorStore
from src.types import ActionType, DisruptionType


@pytest.fixture
def cfg():
    seed_everything(7)
    return load_config()


# ---------- setup flow ---------------------------------------------


def test_setup_flow_produces_dag_and_predictor(cfg):
    result = run_setup(
        syllabus_text="Course on linear algebra and regression.",
        provider=MockProvider(),
        config=cfg,
        today=date(2026, 5, 9),
    )
    result.kg.validate_dag()
    assert {t.id for t in result.kg.topics()} == {"linear_algebra", "regression"}
    # Predictor initialised for every topic.
    for t in result.kg.topics():
        assert result.predictor.has_topic(t.id)
    # Initial plan includes at least one introduce-new session.
    assert any(s.action == ActionType.INTRODUCE_NEW for s in result.initial_plan)


def test_setup_flow_invokes_verifier(cfg):
    seen = {}

    def verifier(kg: KnowledgeGraph) -> KnowledgeGraph:
        seen["count"] = len(kg.topics())
        return kg

    run_setup(
        syllabus_text="x",
        provider=MockProvider(),
        config=cfg,
        today=date(2026, 5, 9),
        verifier=verifier,
    )
    assert seen["count"] >= 1


# ---------- weekly flow --------------------------------------------


def _setup_state(cfg):
    """Common setup: KG + predictor + store with materials."""
    result = run_setup(
        syllabus_text="x", provider=MockProvider(), config=cfg, today=date(2026, 5, 9)
    )
    store = InMemoryVectorStore()
    return result.kg, result.predictor, store


def test_weekly_flow_ingests_and_summarises(cfg):
    kg, predictor, store = _setup_state(cfg)
    result = run_weekly(
        week_number=1,
        materials=[
            ("linear_algebra", "Vectors are tuples of numbers " * 50),
            ("regression", "Regression fits a line " * 50),
        ],
        kg=kg,
        predictor=predictor,
        store=store,
        provider=MockProvider(),
        mastery_diff={"regression": (0.1, 0.4)},
        sessions_log="(integration test)",
    )
    assert result.chunks_added > 0
    assert isinstance(result.summary, str) and result.summary
    assert len(store) > 0


def test_weekly_flow_rejects_unknown_topic(cfg):
    kg, predictor, store = _setup_state(cfg)
    with pytest.raises(KeyError):
        run_weekly(
            week_number=1,
            materials=[("missing_topic", "body")],
            kg=kg,
            predictor=predictor,
            store=store,
            provider=MockProvider(),
            mastery_diff={},
            sessions_log="",
        )


# ---------- session flow ------------------------------------------


def test_session_flow_updates_mastery(cfg):
    kg, predictor, store = _setup_state(cfg)
    # Ingest some material so focused-RAG can return chunks.
    run_weekly(
        week_number=1,
        materials=[("regression", "Regression fits a line " * 30)],
        kg=kg,
        predictor=predictor,
        store=store,
        provider=MockProvider(),
        mastery_diff={},
        sessions_log="",
    )

    before = predictor.mastery("regression")

    def perfect_answer(q):
        return q.answer

    result = run_session(
        topic_id="regression",
        kg=kg,
        predictor=predictor,
        store=store,
        provider=MockProvider(),
        answer_fn=perfect_answer,
        config=cfg,
    )
    after = predictor.mastery("regression")

    assert len(result.graded) >= 1
    assert all(g.correct for g in result.graded)
    assert after >= before  # belief should not decrease on perfect answers
    assert result.explanation
    assert result.next_action in {a for a in ActionType}


# ---------- disruption flow ---------------------------------------


def test_disruption_flow_sick_day_replans(cfg):
    from src.types import Session

    # MockProvider's canned sick_day response uses 2026-05-09 — align the
    # schedule so the replanner has something to push.
    sick_date = date(2026, 5, 9)
    schedule = [
        Session(
            topic_id="a",
            action=ActionType.QUIZ_EXISTING,
            scheduled_date=sick_date,
            duration_minutes=30,
        ),
        Session(
            topic_id="b",
            action=ActionType.QUIZ_EXISTING,
            scheduled_date=sick_date,
            duration_minutes=30,
        ),
    ]
    result = handle_disruption(
        report_text="I was sick today.",
        schedule=schedule,
        provider=MockProvider(),
        config=cfg,
        today=sick_date,
    )
    assert result.update.type == DisruptionType.SICK_DAY
    assert result.confirmation
    # No sessions remain on the sick day.
    assert all(s.scheduled_date != sick_date for s in result.new_schedule)
