"""Grading for MCQ (direct comparison) and short-answer (LLM rubric)."""

from __future__ import annotations

from src.llm.json_utils import parse_json_response
from src.llm.provider import LLMProvider
from src.types import GradedResponse, QuestionType, QuizResponse


def grade_mcq(response: QuizResponse) -> GradedResponse:
    if response.question.type != QuestionType.MCQ:
        raise ValueError("grade_mcq received non-MCQ question")
    correct = _normalise(response.student_answer) == _normalise(response.question.answer)
    return GradedResponse(
        response=response,
        score=1.0 if correct else 0.0,
        correct=correct,
        feedback="Correct." if correct else f"Expected: {response.question.answer}",
    )


def grade_short_response(response: QuizResponse, provider: LLMProvider) -> GradedResponse:
    if response.question.type != QuestionType.SHORT:
        raise ValueError("grade_short_response received non-short question")
    raw = provider.render_and_complete(
        "GRADING_FREE_RESPONSE",
        {
            "question": response.question.stem,
            "reference_answer": response.question.answer,
            "student_answer": response.student_answer,
        },
    )
    data = parse_json_response(raw)
    score = float(data["score"])
    return GradedResponse(
        response=response,
        score=score,
        correct=score >= 0.5,
        feedback=data.get("feedback", ""),
    )


def _normalise(s: str) -> str:
    return s.strip().lower()
