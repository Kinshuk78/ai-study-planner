"""Session flow.

focused RAG -> LLM quiz generation -> student responses -> grading
(MCQ direct, free-response via LLM) -> BKT updates -> Q-learning state
update -> next session -> LLM explanation via graph-traversal RAG.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.bkt import BKTPredictor
from src.kg import KnowledgeGraph
from src.llm.provider import LLMProvider
from src.llm.tasks import explain, generate_quiz
from src.llm.tasks.grader import grade_mcq, grade_short_response
from src.rag import Embedder, focused_retrieve, graph_traversal_retrieve
from src.rag.vectorstore import VectorStore
from src.scheduler.q_learning import QLearningAgent
from src.scheduler.rules import eligible_actions
from src.types import (
    ActionType,
    GradedResponse,
    QuestionType,
    QuizQuestion,
    QuizResponse,
    SchedulerState,
)

AnswerFn = Callable[[QuizQuestion], str]


@dataclass
class SessionResult:
    graded: list[GradedResponse]
    explanation: str
    next_action: ActionType
    mastery_changes: dict[str, tuple[float, float]]


def run_session(
    *,
    topic_id: str,
    kg: KnowledgeGraph,
    predictor: BKTPredictor,
    store: VectorStore,
    provider: LLMProvider,
    answer_fn: AnswerFn,
    agent: QLearningAgent | None = None,
    state: SchedulerState | None = None,
    config: dict,
    num_questions: int = 2,
) -> SessionResult:
    """Execute one full study session.

    :param answer_fn: callable that takes a question and returns the
        student's answer string. Streamlit passes a UI callback;
        simulator passes a sampling function.
    """
    embedder = Embedder(provider)
    topic = kg.get_topic(topic_id)

    focused_chunks = focused_retrieve(
        query=topic.name, topic_id=topic_id, store=store, embedder=embedder
    )
    questions = generate_quiz(
        topic_name=topic.name,
        chunks=focused_chunks,
        num_questions=num_questions,
        provider=provider,
    )

    graded: list[GradedResponse] = []
    mastery_changes: dict[str, tuple[float, float]] = {}
    before = predictor.mastery(topic_id)

    for q in questions:
        ans = answer_fn(q)
        response = QuizResponse(question=q, student_answer=ans)
        if q.type == QuestionType.MCQ:
            g = grade_mcq(response)
        else:
            g = grade_short_response(response, provider)
        predictor.observe(topic_id, correct=g.correct)
        graded.append(g)

    after = predictor.mastery(topic_id)
    mastery_changes[topic_id] = (before, after)

    # Graph-traversal explanation grounded in topic + prerequisites.
    traversal_chunks = graph_traversal_retrieve(
        query=topic.name, topic_id=topic_id, kg=kg, store=store, embedder=embedder
    )
    explanation = explain(
        topic_name=topic.name,
        question=f"Summarise key points of {topic.name} for a student at mastery {after:.2f}",
        mastery_level=after,
        chunks=traversal_chunks,
        provider=provider,
    )

    next_action = _choose_next_action(
        kg=kg, predictor=predictor, agent=agent, state=state, config=config
    )

    return SessionResult(
        graded=graded,
        explanation=explanation,
        next_action=next_action,
        mastery_changes=mastery_changes,
    )


def _choose_next_action(
    *,
    kg: KnowledgeGraph,
    predictor: BKTPredictor,
    agent: QLearningAgent | None,
    state: SchedulerState | None,
    config: dict,
) -> ActionType:
    from datetime import date

    bkt = predictor.all_mastery()
    candidates = list(bkt.keys())
    eligible = eligible_actions(
        candidate_topic_ids=candidates,
        bkt_estimates=bkt,
        kg=kg,
        schedule=[],
        today=date.today(),
        config=config,
    )
    if not eligible:
        return ActionType.REST
    if agent is not None and state is not None:
        return agent.select_action(state, eligible, explore=False)
    # Default policy when no agent is supplied: pick the first eligible non-REST.
    for a in eligible:
        if a != ActionType.REST:
            return a
    return ActionType.REST
