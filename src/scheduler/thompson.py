"""Thompson Sampling baseline.

One Beta arm per action type. Used as a non-trivial baseline against
which Q-learning is compared in
``evaluation/scheduler_comparison/``.
"""

from __future__ import annotations

import numpy as np

from src.types import ActionType


class ThompsonSamplingPolicy:
    def __init__(
        self, actions: list[ActionType] | None = None, rng: np.random.Generator | None = None
    ) -> None:
        self.actions = actions or list(ActionType)
        self._alpha = {a: 1.0 for a in self.actions}
        self._beta = {a: 1.0 for a in self.actions}
        self._rng = rng or np.random.default_rng()

    def select_action(self, eligible: list[ActionType]) -> ActionType:
        if not eligible:
            raise ValueError("eligible action set is empty")
        samples = {a: self._rng.beta(self._alpha[a], self._beta[a]) for a in eligible}
        return max(samples, key=lambda a: samples[a])

    def update(self, action: ActionType, reward: float) -> None:
        """Bernoulli-style update: positive reward = success, otherwise failure.

        We binarise the reward at zero so a single Beta posterior suffices.
        """
        if reward > 0.0:
            self._alpha[action] += 1.0
        else:
            self._beta[action] += 1.0

    def expectations(self) -> dict[ActionType, float]:
        return {a: self._alpha[a] / (self._alpha[a] + self._beta[a]) for a in self.actions}
