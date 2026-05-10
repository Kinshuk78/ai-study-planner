"""Hard feasibility layer.

This is the rule layer of the decomposed scheduler:

* **Rules** (this file) emit the *eligible set* — actions that satisfy
  every hard constraint (prerequisites, capacity, deadline reachability).
* **Q-learning** chooses among the eligible set.

The decomposition is invariant. Rules are **not** soft preferences and
must not be relaxed by the policy.
"""

from __future__ import annotations

from datetime import date

from src.kg import KnowledgeGraph
from src.types import ActionType, Session


def is_prerequisite_met(
    topic_id: str,
    bkt_estimates: dict[str, float],
    kg: KnowledgeGraph,
    mastery_threshold: float,
) -> bool:
    """All direct prerequisites of ``topic_id`` are at or above threshold."""
    for prereq in kg.get_prerequisites(topic_id):
        if bkt_estimates.get(prereq, 0.0) < mastery_threshold:
            return False
    return True


def is_capacity_ok(
    candidate_session: Session,
    current_schedule: list[Session],
    daily_capacity_minutes: int,
    weekly_capacity_minutes: int,
) -> bool:
    """The session fits within the day's and the week's remaining capacity."""
    same_day = [s for s in current_schedule if s.scheduled_date == candidate_session.scheduled_date]
    day_used = sum(s.duration_minutes for s in same_day)
    if day_used + candidate_session.duration_minutes > daily_capacity_minutes:
        return False

    week_start = _week_start(candidate_session.scheduled_date)
    same_week = [s for s in current_schedule if _week_start(s.scheduled_date) == week_start]
    week_used = sum(s.duration_minutes for s in same_week)
    return week_used + candidate_session.duration_minutes <= weekly_capacity_minutes


def is_deadline_reachable(
    remaining_topics: int,
    days_left: int,
    daily_capacity_minutes: int,
    session_duration_minutes: int,
) -> bool:
    """At least one feasible schedule exists for the remaining topics."""
    if remaining_topics <= 0:
        return True
    if days_left <= 0:
        return False
    sessions_per_day = max(1, daily_capacity_minutes // session_duration_minutes)
    achievable_sessions = days_left * sessions_per_day
    return achievable_sessions >= remaining_topics


def eligible_actions(
    *,
    candidate_topic_ids: list[str],
    bkt_estimates: dict[str, float],
    kg: KnowledgeGraph,
    schedule: list[Session],
    today: date,
    config: dict,
) -> list[ActionType]:
    """Compute the eligible action set for the next session.

    The eligible set is the subset of all four action types that have at
    least one feasible target topic given the current state. ``REST`` is
    always eligible. Q-learning chooses among the returned actions.
    """
    sched = config["scheduler"]
    bkt_cfg = config["bkt"]
    daily = sched["daily_capacity_minutes"]
    weekly = sched["weekly_capacity_minutes"]
    duration = sched["session_duration_minutes"]
    mastery_threshold = bkt_cfg["mastery_threshold"]
    at_risk_threshold = bkt_cfg["at_risk_threshold"]

    capacity_probe = Session(
        topic_id=None,
        action=ActionType.REST,
        scheduled_date=today,
        duration_minutes=duration,
    )
    has_capacity = is_capacity_ok(capacity_probe, schedule, daily, weekly)

    eligible: list[ActionType] = [ActionType.REST]
    if not has_capacity:
        return eligible

    # INTRODUCE_NEW: any candidate not yet mastered whose prereqs are met.
    for tid in candidate_topic_ids:
        mastery = bkt_estimates.get(tid, 0.0)
        if mastery >= mastery_threshold:
            continue
        if not is_prerequisite_met(tid, bkt_estimates, kg, mastery_threshold):
            continue
        eligible.append(ActionType.INTRODUCE_NEW)
        break

    # REVIEW_WEAKEST: at least one introduced topic below at_risk_threshold.
    introduced = [tid for tid in candidate_topic_ids if tid in bkt_estimates]
    if any(bkt_estimates[tid] < at_risk_threshold for tid in introduced):
        eligible.append(ActionType.REVIEW_WEAKEST)

    # QUIZ_EXISTING: at least one introduced topic between thresholds.
    if any(at_risk_threshold <= bkt_estimates[tid] < mastery_threshold for tid in introduced):
        eligible.append(ActionType.QUIZ_EXISTING)

    return eligible


# ----- helpers ------------------------------------------------------


def _week_start(d: date) -> date:
    """Return the Monday of the ISO week containing ``d``."""
    from datetime import timedelta

    return d - timedelta(days=d.weekday())
