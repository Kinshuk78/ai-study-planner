"""State discretisation for tabular Q-learning.

Raw state is the four-tuple
``(fraction_mastered, days_remaining, num_at_risk, last_action)``.
The encoder collapses it into an ``int`` state id by binning the
continuous / unbounded coordinates.

Total state space size = ``F * D * R * A`` where
``F = bins.fraction_mastered``, ``D = bins.days_remaining``,
``R = bins.num_at_risk``, ``A = |actions|``. Default config gives
``5 * 5 * 4 * 4 = 400``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.types import ActionType, SchedulerState


@dataclass(frozen=True)
class StateEncoder:
    fraction_mastered_bins: int
    days_remaining_bins: int
    num_at_risk_bins: int
    max_days_remaining: int
    max_at_risk: int
    actions: tuple[ActionType, ...]

    @classmethod
    def from_config(cls, config: dict) -> StateEncoder:
        bins = config["q_learning"]["state_bins"]
        actions = tuple(ActionType(a) for a in config["scheduler"]["actions"])
        # Defaults: 1 semester ≈ 14 weeks ≈ 100 days; ~30 topics.
        max_days = config["q_learning"].get("max_days_remaining", 100)
        max_at_risk = config["q_learning"].get("max_at_risk", 20)
        return cls(
            fraction_mastered_bins=bins["fraction_mastered"],
            days_remaining_bins=bins["days_remaining"],
            num_at_risk_bins=bins["num_at_risk"],
            max_days_remaining=max_days,
            max_at_risk=max_at_risk,
            actions=actions,
        )

    @property
    def num_states(self) -> int:
        return (
            self.fraction_mastered_bins
            * self.days_remaining_bins
            * self.num_at_risk_bins
            * len(self.actions)
        )

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    # ---- encoding ---------------------------------------------------

    def encode(self, state: SchedulerState) -> int:
        f = _bin_fraction(state.fraction_mastered, self.fraction_mastered_bins)
        d = _bin_value(state.days_remaining, 0, self.max_days_remaining, self.days_remaining_bins)
        r = _bin_value(state.num_at_risk, 0, self.max_at_risk, self.num_at_risk_bins)
        a = self.actions.index(state.last_action)
        idx = (((f * self.days_remaining_bins) + d) * self.num_at_risk_bins + r) * len(
            self.actions
        ) + a
        return idx

    def decode(self, state_id: int) -> dict:
        a = state_id % len(self.actions)
        s = state_id // len(self.actions)
        r = s % self.num_at_risk_bins
        s //= self.num_at_risk_bins
        d = s % self.days_remaining_bins
        f = s // self.days_remaining_bins
        return {
            "fraction_mastered_bin": f,
            "days_remaining_bin": d,
            "num_at_risk_bin": r,
            "last_action": self.actions[a].value,
        }

    def action_index(self, action: ActionType) -> int:
        return self.actions.index(action)


# ---------- binning helpers -----------------------------------------


def _bin_fraction(x: float, num_bins: int) -> int:
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    idx = int(x * num_bins)
    return min(idx, num_bins - 1)


def _bin_value(x: int, lo: int, hi: int, num_bins: int) -> int:
    if x <= lo:
        return 0
    if x >= hi:
        return num_bins - 1
    span = hi - lo
    idx = int((x - lo) / span * num_bins)
    return min(idx, num_bins - 1)
