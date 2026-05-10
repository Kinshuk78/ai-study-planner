"""Tabular Q-learning over the discretised scheduler state.

Why tabular (not DQN):

* The discretised state is small (~300-400 cells).
* The Q-table is **inspectable** for the report — we can show what each
  cell prefers.
* It is honestly defensible from the course content.

The policy is **constrained** to the eligible action set produced by
the rule layer; Q-learning never overrides feasibility.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.scheduler.state import StateEncoder
from src.types import ActionType, SchedulerState


@dataclass
class QLearningHyperparams:
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    num_training_episodes: int

    @classmethod
    def from_config(cls, config: dict) -> QLearningHyperparams:
        c = config["q_learning"]
        return cls(
            alpha=c["alpha"],
            gamma=c["gamma"],
            epsilon_start=c["epsilon_start"],
            epsilon_end=c["epsilon_end"],
            epsilon_decay=c["epsilon_decay"],
            num_training_episodes=c["num_training_episodes"],
        )


def _empty_q_table() -> np.ndarray:
    """Sentinel default replaced in :meth:`QLearningAgent.__post_init__`."""
    return np.zeros((0, 0), dtype=np.float64)


@dataclass
class QLearningAgent:
    encoder: StateEncoder
    hyperparams: QLearningHyperparams
    q_table: np.ndarray = field(default_factory=_empty_q_table)
    epsilon: float = -1.0  # sentinel replaced in __post_init__

    def __post_init__(self) -> None:
        if self.q_table.size == 0:
            self.q_table = np.zeros(
                (self.encoder.num_states, self.encoder.num_actions), dtype=np.float64
            )
        if self.epsilon < 0.0:
            self.epsilon = self.hyperparams.epsilon_start

    # ---- action selection -----------------------------------------

    def select_action(
        self, state: SchedulerState, eligible: list[ActionType], explore: bool = True
    ) -> ActionType:
        """Epsilon-greedy over the eligible set.

        :param explore: if False, always exploits (greedy). Used at
            evaluation time.
        """
        if not eligible:
            raise ValueError("eligible action set is empty")
        if explore and random.random() < self.epsilon:
            return random.choice(eligible)

        sid = self.encoder.encode(state)
        eligible_indices = [self.encoder.action_index(a) for a in eligible]
        q_values = self.q_table[sid, eligible_indices]
        best_local = int(np.argmax(q_values))
        return eligible[best_local]

    # ---- update ----------------------------------------------------

    def update(
        self,
        state: SchedulerState,
        action: ActionType,
        reward: float,
        next_state: SchedulerState,
        next_eligible: list[ActionType],
        done: bool,
    ) -> None:
        sid = self.encoder.encode(state)
        aid = self.encoder.action_index(action)
        nsid = self.encoder.encode(next_state)

        if done or not next_eligible:
            target = reward
        else:
            next_indices = [self.encoder.action_index(a) for a in next_eligible]
            target = reward + self.hyperparams.gamma * float(
                np.max(self.q_table[nsid, next_indices])
            )

        old = self.q_table[sid, aid]
        self.q_table[sid, aid] = old + self.hyperparams.alpha * (target - old)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.hyperparams.epsilon_end,
            self.epsilon * self.hyperparams.epsilon_decay,
        )

    # ---- persistence ----------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.q_table)
        meta = {
            "epsilon": self.epsilon,
            "num_states": self.encoder.num_states,
            "num_actions": self.encoder.num_actions,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.q_table = np.load(path.with_suffix(".npy"))
        meta = json.loads(path.with_suffix(".json").read_text())
        self.epsilon = meta["epsilon"]


# ---- reward function ----------------------------------------------


def compute_reward(
    *,
    mastery_gain: float,
    on_track: bool,
    deadline_missed: bool,
    num_at_risk: int,
    config: dict,
) -> float:
    """Reward shaping shared by the trainer and the simulator."""
    cfg = config["q_learning"]["reward"]
    r = cfg["mastery_gain_weight"] * mastery_gain
    if on_track:
        r += cfg["on_track_bonus"]
    if deadline_missed:
        r += cfg["deadline_miss_penalty"]
    r += cfg["at_risk_penalty"] * num_at_risk
    return float(r)
