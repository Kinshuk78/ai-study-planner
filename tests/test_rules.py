from __future__ import annotations

from datetime import date

import pytest

from src.config import load_config
from src.kg import KnowledgeGraph, Topic
from src.scheduler.rules import (
    eligible_actions,
    is_capacity_ok,
    is_deadline_reachable,
    is_prerequisite_met,
    select_topic_for_action,
)
from src.types import ActionType, Session


@pytest.fixture
def kg() -> KnowledgeGraph:
    g = KnowledgeGraph()
    for tid in ("a", "b", "c"):
        g.add_topic(Topic(id=tid, name=tid))
    g.add_prerequisite("a", "b")
    g.add_prerequisite("b", "c")
    return g


def session(d: date, minutes: int = 30) -> Session:
    return Session(
        topic_id="x", action=ActionType.QUIZ_EXISTING, scheduled_date=d, duration_minutes=minutes
    )


# ---------- prerequisite gate ---------------------------------------


def test_prerequisite_met_when_above_threshold(kg):
    bkt = {"a": 0.9, "b": 0.4}
    assert is_prerequisite_met("b", bkt, kg, mastery_threshold=0.8)


def test_prerequisite_blocked_when_below_threshold(kg):
    bkt = {"a": 0.5}
    assert not is_prerequisite_met("b", bkt, kg, mastery_threshold=0.8)


def test_root_topic_always_eligible(kg):
    assert is_prerequisite_met("a", {}, kg, mastery_threshold=0.8)


# ---------- capacity gate -------------------------------------------


def test_daily_capacity_blocks():
    today = date(2026, 5, 11)  # Monday
    schedule = [session(today, 60), session(today, 30)]
    candidate = session(today, 30)
    assert not is_capacity_ok(
        candidate, schedule, daily_capacity_minutes=90, weekly_capacity_minutes=540
    )


def test_daily_capacity_passes():
    today = date(2026, 5, 11)
    schedule = [session(today, 30)]
    candidate = session(today, 30)
    assert is_capacity_ok(
        candidate, schedule, daily_capacity_minutes=90, weekly_capacity_minutes=540
    )


def test_weekly_capacity_blocks():
    monday = date(2026, 5, 11)
    schedule = [session(monday, 90)] * 5  # 450 mins on Monday alone is fine for week
    candidate = session(date(2026, 5, 16), 120)  # Saturday — weekly cap 540
    assert not is_capacity_ok(
        candidate, schedule, daily_capacity_minutes=180, weekly_capacity_minutes=540
    )


# ---------- deadline reachability -----------------------------------


def test_deadline_reachable_with_buffer():
    assert is_deadline_reachable(
        remaining_topics=4, days_left=4, daily_capacity_minutes=90, session_duration_minutes=30
    )


def test_deadline_unreachable_when_no_days_left():
    assert not is_deadline_reachable(
        remaining_topics=2, days_left=0, daily_capacity_minutes=90, session_duration_minutes=30
    )


def test_deadline_unreachable_when_too_many_topics():
    assert not is_deadline_reachable(
        remaining_topics=20, days_left=2, daily_capacity_minutes=30, session_duration_minutes=30
    )


def test_deadline_trivially_true_when_nothing_remains():
    assert is_deadline_reachable(0, 0, 90, 30)


# ---------- eligible-actions integration ----------------------------


def test_eligible_actions_includes_rest_only_when_no_capacity(kg):
    cfg = load_config()
    today = date(2026, 5, 11)
    full_day = [
        Session(
            topic_id="x",
            action=ActionType.QUIZ_EXISTING,
            scheduled_date=today,
            duration_minutes=cfg["scheduler"]["daily_capacity_minutes"],
        ),
    ]
    actions = eligible_actions(
        candidate_topic_ids=["a"],
        bkt_estimates={"a": 0.1},
        kg=kg,
        schedule=full_day,
        today=today,
        config=cfg,
    )
    assert actions == [ActionType.REST]


def test_eligible_actions_introduce_new_when_prereqs_met(kg):
    cfg = load_config()
    today = date(2026, 5, 11)
    bkt = {"a": 0.9}  # 'b' becomes eligible
    actions = eligible_actions(
        candidate_topic_ids=["b"],
        bkt_estimates=bkt,
        kg=kg,
        schedule=[],
        today=today,
        config=cfg,
    )
    assert ActionType.INTRODUCE_NEW in actions
    assert ActionType.REST in actions


def test_eligible_actions_blocks_introduce_when_prereqs_missing(kg):
    cfg = load_config()
    today = date(2026, 5, 11)
    actions = eligible_actions(
        candidate_topic_ids=["b"],
        bkt_estimates={"a": 0.2},
        kg=kg,
        schedule=[],
        today=today,
        config=cfg,
    )
    assert ActionType.INTRODUCE_NEW not in actions


def test_eligible_actions_review_weakest_when_at_risk(kg):
    cfg = load_config()
    today = date(2026, 5, 11)
    bkt = {"a": 0.2}  # introduced, below at_risk_threshold (default 0.5)
    actions = eligible_actions(
        candidate_topic_ids=["a"],
        bkt_estimates=bkt,
        kg=kg,
        schedule=[],
        today=today,
        config=cfg,
    )
    assert ActionType.REVIEW_WEAKEST in actions


def test_eligible_actions_does_not_review_cold_start_topic(kg):
    cfg = load_config()
    today = date(2026, 5, 11)
    bkt = {"a": cfg["bkt"]["priors"]["L0"]}
    actions = eligible_actions(
        candidate_topic_ids=["a"],
        bkt_estimates=bkt,
        kg=kg,
        schedule=[],
        today=today,
        config=cfg,
    )
    assert ActionType.INTRODUCE_NEW in actions
    assert ActionType.REVIEW_WEAKEST not in actions
    assert ActionType.QUIZ_EXISTING not in actions


def test_eligible_actions_does_not_reintroduce_observed_topic(kg):
    cfg = load_config()
    today = date(2026, 5, 11)
    bkt = {"a": 0.2}
    actions = eligible_actions(
        candidate_topic_ids=["a"],
        bkt_estimates=bkt,
        kg=kg,
        schedule=[],
        today=today,
        config=cfg,
    )
    assert ActionType.INTRODUCE_NEW not in actions
    assert ActionType.REVIEW_WEAKEST in actions


def test_eligible_actions_quiz_when_between_thresholds(kg):
    cfg = load_config()
    today = date(2026, 5, 11)
    bkt = {"a": 0.6}  # between 0.5 and 0.8
    actions = eligible_actions(
        candidate_topic_ids=["a"],
        bkt_estimates=bkt,
        kg=kg,
        schedule=[],
        today=today,
        config=cfg,
    )
    assert ActionType.QUIZ_EXISTING in actions


def test_select_topic_for_action_respects_cold_start_lifecycle(kg):
    cfg = load_config()
    topic_ids = ["a", "b", "c"]
    bkt = {tid: cfg["bkt"]["priors"]["L0"] for tid in topic_ids}
    assert (
        select_topic_for_action(
            ActionType.INTRODUCE_NEW,
            candidate_topic_ids=topic_ids,
            bkt_estimates=bkt,
            kg=kg,
            schedule=[],
            config=cfg,
        )
        == "a"
    )
    assert (
        select_topic_for_action(
            ActionType.REVIEW_WEAKEST,
            candidate_topic_ids=topic_ids,
            bkt_estimates=bkt,
            kg=kg,
            schedule=[],
            config=cfg,
        )
        is None
    )


def test_select_topic_for_action_respects_prerequisites(kg):
    cfg = load_config()
    topic_ids = ["a", "b", "c"]
    bkt = {"a": 0.9, "b": cfg["bkt"]["priors"]["L0"], "c": cfg["bkt"]["priors"]["L0"]}
    assert (
        select_topic_for_action(
            ActionType.INTRODUCE_NEW,
            candidate_topic_ids=topic_ids,
            bkt_estimates=bkt,
            kg=kg,
            schedule=[],
            config=cfg,
        )
        == "b"
    )
