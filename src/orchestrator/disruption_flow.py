"""Disruption flow.

free-text self-report -> LLM parses to structured update
(sick_day / deadline_change / capacity_change / completed_externally)
-> scheduler reflow -> LLM confirmation message.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.llm.provider import LLMProvider
from src.llm.tasks import parse_disruption
from src.scheduler.replanner import replan
from src.types import DisruptionType, DisruptionUpdate, Session


@dataclass
class DisruptionResult:
    update: DisruptionUpdate
    new_schedule: list[Session]
    confirmation: str


def handle_disruption(
    *,
    report_text: str,
    schedule: list[Session],
    provider: LLMProvider,
    config: dict,
    today: date | None = None,
) -> DisruptionResult:
    today = today or date.today()
    update = parse_disruption(report_text, provider, today=today)
    new_schedule = replan(schedule, update, today=today, config=config)
    confirmation = _format_confirmation(update, new_schedule)
    return DisruptionResult(update=update, new_schedule=new_schedule, confirmation=confirmation)


def _format_confirmation(update: DisruptionUpdate, new_schedule: list[Session]) -> str:
    """Deterministic confirmation message — no LLM needed; the parsed
    update is already structured."""
    if update.type == DisruptionType.SICK_DAY:
        return (
            f"Recorded a sick day on {update.payload['date']}. Affected sessions "
            f"have been pushed to the next available slots ({len(new_schedule)} sessions remain)."
        )
    if update.type == DisruptionType.DEADLINE_CHANGE:
        return (
            f"Updated deadline to {update.payload['new_deadline']}. "
            f"Schedule re-packed to fit the new window ({len(new_schedule)} sessions)."
        )
    if update.type == DisruptionType.CAPACITY_CHANGE:
        return (
            f"Updated daily capacity to {update.payload['new_daily_minutes']} minutes. "
            f"Schedule re-flowed under the new capacity ({len(new_schedule)} sessions)."
        )
    if update.type == DisruptionType.COMPLETED_EXTERNALLY:
        ids = ", ".join(update.payload.get("topic_ids", []))
        return (
            f"Marked '{ids}' as completed externally. "
            f"Removed from upcoming sessions ({len(new_schedule)} sessions remain)."
        )
    return f"Applied disruption: {update.type.value}."  # pragma: no cover
