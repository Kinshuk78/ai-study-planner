"""Setup flow.

syllabus -> LLM-extracted draft KG -> human verification -> save kg.json
-> BKT init -> week-1 plan.

The orchestrator is a deterministic coordinator, not an agent. There is
no LLM-driven decision making about flow ordering.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.bkt import BKTPredictor, default_params
from src.kg import KnowledgeGraph
from src.llm.provider import LLMProvider
from src.llm.tasks import extract_kg
from src.scheduler.rules import eligible_actions
from src.types import ActionType, Session


@dataclass
class SetupResult:
    kg: KnowledgeGraph
    predictor: BKTPredictor
    initial_plan: list[Session]


KGVerifier = Callable[[KnowledgeGraph], KnowledgeGraph]


def run_setup(
    *,
    syllabus_text: str,
    provider: LLMProvider,
    config: dict,
    today: date,
    save_kg_to: str | Path | None = None,
    verifier: KGVerifier | None = None,
) -> SetupResult:
    """Run the setup flow end-to-end.

    :param verifier: optional human-in-the-loop callback that receives
        the draft KG and returns the verified KG. The Streamlit UI
        passes a callback; tests pass an identity function.
    """
    draft_kg = extract_kg(syllabus_text, provider)
    verified_kg = verifier(draft_kg) if verifier is not None else draft_kg
    verified_kg.validate_dag()

    if save_kg_to is not None:
        verified_kg.save(save_kg_to)

    predictor = BKTPredictor(default_params())
    for topic in verified_kg.topics():
        predictor.init_topic(topic.id)

    plan = _build_initial_plan(verified_kg, predictor, config, today)
    return SetupResult(kg=verified_kg, predictor=predictor, initial_plan=plan)


def _build_initial_plan(
    kg: KnowledgeGraph,
    predictor: BKTPredictor,
    config: dict,
    today: date,
) -> list[Session]:
    """Greedy first-week plan: walk topology and place sessions on the
    earliest day that admits an INTRODUCE_NEW action."""
    sched_cfg = config["scheduler"]
    duration = sched_cfg["session_duration_minutes"]
    plan: list[Session] = []
    candidate_topics = kg.topological_order()
    bkt = predictor.all_mastery()

    current_date = today
    for tid in candidate_topics:
        actions = eligible_actions(
            candidate_topic_ids=[tid],
            bkt_estimates=bkt,
            kg=kg,
            schedule=plan,
            today=current_date,
            config=config,
        )
        if ActionType.INTRODUCE_NEW not in actions:
            # Day is full or prereq not met yet — advance a day.
            from datetime import timedelta

            current_date = current_date + timedelta(days=1)
            continue
        plan.append(
            Session(
                topic_id=tid,
                action=ActionType.INTRODUCE_NEW,
                scheduled_date=current_date,
                duration_minutes=duration,
            )
        )
    return plan
