"""Exponential forgetting-curve decay applied to BKT mastery.

We decay mastery toward zero with half-life ``half_life_days``:

    mastery(t + dt) = mastery(t) * 0.5 ** (dt / half_life_days)

Multiplicative form keeps the result in ``[0, 1]`` automatically.
"""

from __future__ import annotations


def decay_mastery(p_mastery: float, days_elapsed: float, half_life_days: float) -> float:
    """Applies exponential decay over ``days_elapsed`` days.

    :raises ValueError: if ``half_life_days`` is non-positive or
        ``days_elapsed`` is negative.
    """
    if half_life_days <= 0.0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    if days_elapsed < 0.0:
        raise ValueError(f"days_elapsed must be non-negative, got {days_elapsed}")
    if not 0.0 <= p_mastery <= 1.0:
        raise ValueError(f"p_mastery must be in [0, 1], got {p_mastery}")

    factor = 0.5 ** (days_elapsed / half_life_days)
    decayed = float(p_mastery * factor)

    # Numerically guard the bounds.
    if decayed < 0.0:
        return 0.0
    if decayed > 1.0:
        return 1.0
    return decayed
