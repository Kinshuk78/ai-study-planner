"""BKT predictor — standard HMM update.

The predictor exposes a stateless :func:`update` (one observation step)
and a :class:`BKTPredictor` that maintains per-topic mastery estimates
across a sequence of observations and time gaps.
"""

from __future__ import annotations

from src.bkt.decay import decay_mastery
from src.bkt.params import BKTParams


def update(p_mastery: float, correct: bool, params: BKTParams) -> float:
    """Single BKT update step.

    Given the current ``P(mastered)`` and an observation, returns the
    posterior probability of mastery. Output is clipped to ``[0, 1]``.
    """
    if not 0.0 <= p_mastery <= 1.0:
        raise ValueError(f"p_mastery must be in [0, 1], got {p_mastery}")

    # Posterior over the *current* mastery state, conditioned on the observation.
    if correct:
        numerator = p_mastery * (1.0 - params.slip)
        denominator = numerator + (1.0 - p_mastery) * params.guess
    else:
        numerator = p_mastery * params.slip
        denominator = numerator + (1.0 - p_mastery) * (1.0 - params.guess)

    # Degenerate priors (e.g. slip=0 with an incorrect response from a mastered
    # student) leave the denominator at zero — fall back to the prior in that case.
    p_post = p_mastery if denominator <= 0.0 else numerator / denominator

    # Apply transition probability for the learning opportunity.
    p_next = p_post + (1.0 - p_post) * params.transit
    return _clip01(p_next)


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class BKTPredictor:
    """Per-topic mastery tracker.

    Use :meth:`init_topic` for each topic, then :meth:`observe` for each
    quiz response and :meth:`apply_decay` between sessions.
    """

    def __init__(self, params: BKTParams) -> None:
        self.params = params
        self._mastery: dict[str, float] = {}

    # ----- topic management ----------------------------------------

    def init_topic(self, topic_id: str, prior: float | None = None) -> None:
        if topic_id in self._mastery:
            return
        self._mastery[topic_id] = self.params.L0 if prior is None else _clip01(prior)

    def has_topic(self, topic_id: str) -> bool:
        return topic_id in self._mastery

    def mastery(self, topic_id: str) -> float:
        if topic_id not in self._mastery:
            raise KeyError(f"unknown topic '{topic_id}' — call init_topic first")
        return self._mastery[topic_id]

    def all_mastery(self) -> dict[str, float]:
        return dict(self._mastery)

    # ----- updates --------------------------------------------------

    def observe(self, topic_id: str, correct: bool) -> float:
        """Records a quiz response and returns the new mastery estimate."""
        prior = self.mastery(topic_id)
        posterior = update(prior, correct, self.params)
        self._mastery[topic_id] = posterior
        return posterior

    def apply_decay(self, topic_id: str, days_elapsed: float, half_life_days: float) -> float:
        """Applies forgetting decay to a single topic."""
        prior = self.mastery(topic_id)
        decayed = decay_mastery(prior, days_elapsed, half_life_days)
        self._mastery[topic_id] = decayed
        return decayed

    def apply_decay_all(self, days_elapsed: float, half_life_days: float) -> None:
        for topic_id in list(self._mastery):
            self.apply_decay(topic_id, days_elapsed, half_life_days)
