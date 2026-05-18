"""Simulated learner.

Each student carries:
  - a *true* internal mastery state per topic (hidden from the planner),
  - a profile-modulated :class:`BKTParams`,
  - a :class:`BKTPredictor` representing the planner's *belief* about
    them.

The simulator updates the true state on each observation and lets the
planner observe a sampled response. Important: the simulator is a
model, not the world; results modelled here do not generalise to real
learners without further validation.
"""

from __future__ import annotations

import numpy as np

from src.bkt import BKTParams, BKTPredictor
from src.bkt.decay import decay_mastery
from src.simulator.profiles import ProfileSpec, apply_profile


class SimulatedStudent:
    def __init__(
        self,
        *,
        student_id: str,
        profile: ProfileSpec,
        base_params: BKTParams,
        rng: np.random.Generator,
    ) -> None:
        self.id = student_id
        self.profile = profile
        self.params = apply_profile(base_params, profile)
        self._true_mastery: dict[str, float] = {}
        self._rng = rng
        self.predictor = BKTPredictor(self.params)

    # ----- topic management -----------------------------------------

    def init_topic(self, topic_id: str) -> None:
        if topic_id not in self._true_mastery:
            self._true_mastery[topic_id] = self.params.L0
        self.predictor.init_topic(topic_id)

    def true_mastery(self, topic_id: str) -> float:
        return self._true_mastery[topic_id]

    def all_true_mastery(self) -> dict[str, float]:
        return dict(self._true_mastery)

    # ----- core dynamics --------------------------------------------

    def p_correct(self, topic_id: str) -> float:
        m = self._true_mastery[topic_id]
        return m * (1.0 - self.params.slip) + (1.0 - m) * self.params.guess

    def respond(self, topic_id: str) -> bool:
        """Sample a response and update **both** the true state and the
        predictor's belief."""
        p = self.p_correct(topic_id)
        correct = bool(self._rng.random() < p)

        # True learning step: with prob `transit`, mastery transitions.
        if self._true_mastery[topic_id] < 1.0 and self._rng.random() < self.params.transit:
            self._true_mastery[topic_id] = 1.0

        # Planner's observation: pure BKT update on belief.
        self.predictor.observe(topic_id, correct)

        return correct

    def apply_decay(self, days_elapsed: float, half_life_days: float) -> None:
        for tid in list(self._true_mastery):
            self._true_mastery[tid] = decay_mastery(
                self._true_mastery[tid],
                days_elapsed,
                half_life_days,
                floor=self.params.L0,
            )
        self.predictor.apply_decay_all(days_elapsed, half_life_days)
