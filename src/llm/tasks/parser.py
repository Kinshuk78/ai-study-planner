"""Parse free-text learner self-reports into a structured update."""

from __future__ import annotations

from datetime import date

from src.llm.json_utils import parse_json_response
from src.llm.provider import LLMProvider
from src.types import DisruptionType, DisruptionUpdate


def parse_disruption(
    report_text: str, provider: LLMProvider, today: date | None = None
) -> DisruptionUpdate:
    today = today or date.today()
    raw = provider.render_and_complete(
        "DISRUPTION_PARSING",
        {"today": today.isoformat(), "report_text": report_text},
    )
    data = parse_json_response(raw)
    dtype = DisruptionType(data["type"])
    confidence = float(data.pop("confidence", 1.0))
    payload = {k: v for k, v in data.items() if k != "type"}
    return DisruptionUpdate(type=dtype, payload=payload, confidence=confidence)
