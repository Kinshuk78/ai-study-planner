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
from src.scheduler.q_learning import QLearningAgent, compute_reward
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
    current_action: ActionType = ActionType.QUIZ_EXISTING,
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

    if agent is not None and state is not None:
        _update_policy_after_session(
            agent=agent,
            state=state,
            action=current_action,
            predictor=predictor,
            mastery_gain=after - before,
            kg=kg,
            config=config,
        )

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


def _update_policy_after_session(
    *,
    agent: QLearningAgent,
    state: SchedulerState,
    action: ActionType,
    predictor: BKTPredictor,
    mastery_gain: float,
    kg: KnowledgeGraph,
    config: dict,
) -> None:
    """Apply one Q-learning update after a completed study session."""
    from datetime import date

    next_state = _state_from_predictor(
        predictor=predictor,
        last_action=action,
        days_remaining=max(0, state.days_remaining - 1),
        config=config,
    )
    next_bkt = predictor.all_mastery()
    next_eligible = eligible_actions(
        candidate_topic_ids=list(next_bkt.keys()),
        bkt_estimates=next_bkt,
        kg=kg,
        schedule=[],
        today=date.today(),
        config=config,
    )
    reward = compute_reward(
        mastery_gain=mastery_gain,
        on_track=next_state.fraction_mastered >= 0.5,
        deadline_missed=next_state.days_remaining <= 0 and next_state.fraction_mastered < 1.0,
        num_at_risk=next_state.num_at_risk,
        config=config,
    )
    agent.update(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        next_eligible=next_eligible,
        done=next_state.days_remaining <= 0,
    )


def _state_from_predictor(
    *,
    predictor: BKTPredictor,
    last_action: ActionType,
    days_remaining: int,
    config: dict,
) -> SchedulerState:
    bkt = predictor.all_mastery()
    mastery_threshold = config["bkt"]["mastery_threshold"]
    at_risk_threshold = config["bkt"]["at_risk_threshold"]
    total = len(bkt)
    fraction_mastered = (
        sum(1 for value in bkt.values() if value >= mastery_threshold) / total
        if total
        else 0.0
    )
    num_at_risk = sum(1 for value in bkt.values() if value < at_risk_threshold)
    return SchedulerState(
        fraction_mastered=fraction_mastered,
        days_remaining=days_remaining,
        num_at_risk=num_at_risk,
        last_action=last_action,
    )
