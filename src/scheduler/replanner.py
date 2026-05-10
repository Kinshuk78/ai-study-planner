"""Disruption replanner.

Given a current schedule and a :class:`DisruptionUpdate`, produce a new
schedule that respects all hard rules. The replanner does **not**
override the rule layer — it works by removing affected sessions and
re-pushing them onto the next feasible slots.
"""

from __future__ import annotations

from datetime import date, timedelta

from src.scheduler.rules import is_capacity_ok
from src.types import DisruptionType, DisruptionUpdate, Session


def replan(
    schedule: list[Session],
    disruption: DisruptionUpdate,
    *,
    today: date,
    config: dict,
) -> list[Session]:
    """Apply a disruption to ``schedule`` and return the new schedule.

    The original list is not mutated.
    """
    new_schedule = [s for s in schedule]

    if disruption.type == DisruptionType.SICK_DAY:
        sick_date = _parse_date(disruption.payload["date"])
        affected = [s for s in new_schedule if s.scheduled_date == sick_date]
        new_schedule = [s for s in new_schedule if s.scheduled_date != sick_date]
        new_schedule = _push_sessions(affected, new_schedule, sick_date + timedelta(days=1), config)

    elif disruption.type == DisruptionType.DEADLINE_CHANGE:
        new_deadline = _parse_date(disruption.payload["new_deadline"])
        # Drop anything past the new deadline; re-pack remaining into the available window.
        affected = [s for s in new_schedule if s.scheduled_date > new_deadline]
        new_schedule = [s for s in new_schedule if s.scheduled_date <= new_deadline]
        new_schedule = _push_sessions(affected, new_schedule, today, config, deadline=new_deadline)

    elif disruption.type == DisruptionType.CAPACITY_CHANGE:
        new_daily = int(disruption.payload["new_daily_minutes"])
        cfg = {**config, "scheduler": {**config["scheduler"], "daily_capacity_minutes": new_daily}}
        new_schedule = _repack(new_schedule, today, cfg)
        config = cfg  # so subsequent calls reflect new capacity

    elif disruption.type == DisruptionType.COMPLETED_EXTERNALLY:
        completed_ids = set(disruption.payload["topic_ids"])
        new_schedule = [s for s in new_schedule if s.topic_id not in completed_ids]

    else:  # pragma: no cover — DisruptionType is exhaustive
        raise ValueError(f"unknown disruption type: {disruption.type}")

    return new_schedule


# ---------- helpers --------------------------------------------------


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _push_sessions(
    sessions: list[Session],
    schedule: list[Session],
    earliest: date,
    config: dict,
    *,
    deadline: date | None = None,
) -> list[Session]:
    """Re-place ``sessions`` onto the earliest feasible slots after ``earliest``."""
    sched = config["scheduler"]
    daily = sched["daily_capacity_minutes"]
    weekly = sched["weekly_capacity_minutes"]
    new_schedule = list(schedule)
    for session in sessions:
        slot = _find_slot(session, new_schedule, earliest, daily, weekly, deadline)
        if slot is None:
            # Cannot place — drop. Caller is expected to surface infeasibility.
            continue
        new_schedule.append(
            Session(
                topic_id=session.topic_id,
                action=session.action,
                scheduled_date=slot,
                duration_minutes=session.duration_minutes,
            )
        )
    return sorted(new_schedule, key=lambda s: s.scheduled_date)


def _find_slot(
    session: Session,
    schedule: list[Session],
    earliest: date,
    daily: int,
    weekly: int,
    deadline: date | None,
) -> date | None:
    candidate = earliest
    horizon = deadline or earliest + timedelta(days=60)
    while candidate <= horizon:
        probe = Session(
            topic_id=session.topic_id,
            action=session.action,
            scheduled_date=candidate,
            duration_minutes=session.duration_minutes,
        )
        if is_capacity_ok(probe, schedule, daily, weekly):
            return candidate
        candidate += timedelta(days=1)
    return None


def _repack(schedule: list[Session], today: date, config: dict) -> list[Session]:
    """Rebuild the schedule under new capacity limits, preserving relative order."""
    future = sorted(
        [s for s in schedule if s.scheduled_date >= today], key=lambda s: s.scheduled_date
    )
    past = [s for s in schedule if s.scheduled_date < today]
    return _push_sessions(future, past, today, config)
