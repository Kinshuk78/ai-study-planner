"""Robust JSON extraction for LLM responses.

Real Claude/Llama responses occasionally wrap JSON in markdown fences
or pad it with a sentence of prose, even with strict-output instructions.
Every task module that parses JSON should call :func:`parse_json_response`
instead of ``json.loads`` directly.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_response(raw: str) -> Any:
    """Parse a possibly-fenced or prose-padded JSON string.

    Strategy:
      1. If wrapped in ``` fences, extract the fenced block.
      2. Otherwise, slice from the first ``{`` or ``[`` to the matching
         closer.
      3. Fall back to the full string.

    Raises :class:`ValueError` if no valid JSON can be parsed.
    """
    if not raw or not raw.strip():
        raise ValueError("empty LLM response")

    candidates: list[str] = []

    fenced = _FENCE_PATTERN.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())

    sliced = _slice_brackets(raw)
    if sliced:
        candidates.append(sliced)

    candidates.append(raw.strip())

    last_err: Exception | None = None
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
    raise ValueError(f"failed to parse JSON from LLM response: {last_err}\nraw: {raw[:300]}")


def _slice_brackets(raw: str) -> str | None:
    """Return the substring from the first ``{`` or ``[`` to its matching closer."""
    open_idx = -1
    open_ch = ""
    for i, ch in enumerate(raw):
        if ch in "{[":
            open_idx = i
            open_ch = ch
            break
    if open_idx == -1:
        return None
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(open_idx, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return raw[open_idx : i + 1]
    return None
