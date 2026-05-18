"""Exponential forgetting-curve decay applied to BKT mastery.

We decay mastery toward a floor with half-life ``half_life_days``:

    floor + (mastery(t) - floor) * 0.5 ** (dt / half_life_days)

For the deployed predictor, the floor should usually be the learner's
initial mastery prior. That prevents untouched or long-idle topics from
decaying below the model's baseline belief and causing excessive review.
"""

from __future__ import annotations


def decay_mastery(
    p_mastery: float,
    days_elapsed: float,
    half_life_days: float,
    floor: float = 0.0,
) -> float:
    """Applies exponential decay over ``days_elapsed`` days.

    :raises ValueError: if ``half_life_days`` is non-positive or
        ``days_elapsed`` is negative, or if probabilities are outside
        ``[0, 1]``.
    """
    if half_life_days <= 0.0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    if days_elapsed < 0.0:
        raise ValueError(f"days_elapsed must be non-negative, got {days_elapsed}")
    if not 0.0 <= p_mastery <= 1.0:
        raise ValueError(f"p_mastery must be in [0, 1], got {p_mastery}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if floor > p_mastery:
        return p_mastery

    factor = 0.5 ** (days_elapsed / half_life_days)
    decayed = float(floor + (p_mastery - floor) * factor)

    # Numerically guard the bounds.
    if decayed < floor:
        return floor
    if decayed > 1.0:
        return 1.0
    return decayed
