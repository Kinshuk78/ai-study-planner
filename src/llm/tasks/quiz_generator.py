"""Generate a quiz from a topic and a set of retrieved chunks."""

from __future__ import annotations

from src.llm.json_utils import parse_json_response
from src.llm.provider import LLMProvider
from src.types import QuestionType, QuizQuestion, RetrievedChunk


def generate_quiz(
    *,
    topic_name: str,
    chunks: list[RetrievedChunk],
    num_questions: int,
    provider: LLMProvider,
    difficulty: str = "medium",
) -> list[QuizQuestion]:
    context = _format_chunks(chunks)
    raw = provider.render_and_complete(
        "QUIZ_GENERATION",
        {
            "topic_name": topic_name,
            "difficulty": difficulty,
            "context": context,
            "num_questions": num_questions,
        },
    )
    data = parse_json_response(raw)
    questions: list[QuizQuestion] = []
    for q in data["questions"]:
        questions.append(
            QuizQuestion(
                stem=q["stem"],
                answer=q["answer"],
                type=QuestionType(q["type"]),
                choices=q.get("choices", []),
                citations=q.get("citations", []),
            )
        )
    return questions


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for rc in chunks:
        lines.append(f"[{rc.chunk.id}] {rc.chunk.text}")
    return "\n\n".join(lines)
