from __future__ import annotations

import json
from datetime import date

import pytest

from src.llm.mock_provider import MockProvider
from src.llm.tasks import (
    explain,
    extract_kg,
    generate_quiz,
    grade_mcq,
    grade_short_response,
    parse_disruption,
    summarise_week,
)
from src.llm.tasks.explainer import extract_citations
from src.types import (
    Chunk,
    DisruptionType,
    QuestionType,
    QuizQuestion,
    QuizResponse,
    RetrievedChunk,
)


def _chunk(cid: str, text: str = "demo") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=cid, text=text, source="x"), score=1.0)


# ---------- mock provider behaviour --------------------------------


def test_mock_returns_default_for_known_prompt():
    p = MockProvider()
    raw = p.render_and_complete("KG_EXTRACTION", {"syllabus_text": "stub"})
    data = json.loads(raw)
    assert "topics" in data and "edges" in data


def test_mock_records_calls():
    p = MockProvider()
    p.render_and_complete("KG_EXTRACTION", {"syllabus_text": "stub"})
    assert len(p.calls) == 1
    assert "curriculum designer" in p.calls[0]["system"].lower()


def test_mock_explicit_override():
    p = MockProvider()
    # Override before calling.
    p.render_and_complete("KG_EXTRACTION", {"syllabus_text": "x"})
    sys_text = p.calls[0]["system"]
    user_text = p.calls[0]["user"]
    p.set_response(sys_text, user_text, '{"topics": [], "edges": []}')
    raw = p.complete(sys_text, user_text)
    assert json.loads(raw) == {"topics": [], "edges": []}


def test_mock_embedding_is_deterministic():
    p = MockProvider()
    a = p.embed("hello world")
    b = p.embed("hello world")
    c = p.embed("different")
    assert a == b
    assert a != c
    assert all(0.0 <= x <= 1.0 for x in a)


# ---------- KG extraction ------------------------------------------


def test_extract_kg_from_default_response():
    kg = extract_kg("Course on regression and linear algebra.", MockProvider())
    ids = {t.id for t in kg.topics()}
    assert {"linear_algebra", "regression"} <= ids


# ---------- quiz generation ----------------------------------------


def test_generate_quiz_produces_questions():
    chunks = [_chunk("chunk_demo_1"), _chunk("chunk_demo_2")]
    qs = generate_quiz(
        topic_name="Regression", chunks=chunks, num_questions=2, provider=MockProvider()
    )
    assert len(qs) >= 1
    assert all(q.citations for q in qs)


# ---------- grading ------------------------------------------------


def test_grade_mcq_correct():
    q = QuizQuestion(stem="2+2", answer="4", type=QuestionType.MCQ, choices=["3", "4"])
    r = QuizResponse(question=q, student_answer="4")
    g = grade_mcq(r)
    assert g.correct and g.score == 1.0


def test_grade_mcq_incorrect():
    q = QuizQuestion(stem="2+2", answer="4", type=QuestionType.MCQ, choices=["3", "4"])
    r = QuizResponse(question=q, student_answer="3")
    g = grade_mcq(r)
    assert not g.correct and g.score == 0.0


def test_grade_mcq_rejects_short():
    q = QuizQuestion(stem="x", answer="y", type=QuestionType.SHORT)
    with pytest.raises(ValueError):
        grade_mcq(QuizResponse(question=q, student_answer="y"))


def test_grade_short_uses_provider():
    q = QuizQuestion(
        stem="define linearity", answer="additivity + homogeneity", type=QuestionType.SHORT
    )
    r = QuizResponse(question=q, student_answer="line stuff")
    g = grade_short_response(r, MockProvider())
    assert 0.0 <= g.score <= 1.0


# ---------- disruption parsing -------------------------------------


def test_parse_disruption_returns_typed():
    update = parse_disruption("I was sick today.", MockProvider(), today=date(2026, 5, 9))
    assert isinstance(update.type, DisruptionType)
    assert 0.0 <= update.confidence <= 1.0


# ---------- explanation -------------------------------------------


def test_explain_includes_citations():
    text = explain(
        topic_name="Regression",
        question="How does it work?",
        mastery_level=0.5,
        chunks=[_chunk("chunk_demo_1"), _chunk("chunk_demo_2")],
        provider=MockProvider(),
    )
    assert extract_citations(text)


def test_explain_raises_when_provider_returns_no_citations():
    p = MockProvider()
    # Override every signature to a citation-free string.
    real_complete = p.complete

    def fake(system: str, user: str, **kwargs):
        real_complete(system, user, **kwargs)
        return "no citations here"

    p.complete = fake  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        explain(
            topic_name="x",
            question="y",
            mastery_level=0.4,
            chunks=[_chunk("a")],
            provider=p,
        )


# ---------- weekly summary ----------------------------------------


def test_summarise_week_returns_text():
    out = summarise_week(
        week_number=2,
        sessions_log="session log",
        mastery_diff={"a": (0.2, 0.6)},
        chunks=[_chunk("chunk_demo_1")],
        provider=MockProvider(),
    )
    assert isinstance(out, str) and len(out) > 0
