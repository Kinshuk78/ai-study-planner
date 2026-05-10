"""Orchestrates a population of :class:`SimulatedStudent` over weeks.

Used both for Q-learning training (in :mod:`src.scheduler`) and for
the scheduler comparison evaluation. All randomness derives from a
single seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bkt import default_params
from src.config import load_config
from src.kg import KnowledgeGraph
from src.simulator.profiles import sample_profile
from src.simulator.student_model import SimulatedStudent


@dataclass
class StudentLog:
    student_id: str
    profile: str
    true_mastery_history: list[dict[str, float]] = field(default_factory=list)
    belief_mastery_history: list[dict[str, float]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    logs: list[StudentLog]
    seed: int


class SimulationRunner:
    def __init__(self, kg: KnowledgeGraph, seed: int | None = None) -> None:
        cfg = load_config()
        self.kg = kg
        self.seed = seed if seed is not None else int(cfg["random_seed"])
        self.rng = np.random.default_rng(self.seed)
        self.num_students = cfg["simulator"]["num_students"]
        self.num_weeks = cfg["simulator"]["num_weeks"]
        self.half_life = cfg["bkt"]["decay"]["half_life_days"]

    def make_students(self) -> list[SimulatedStudent]:
        base = default_params()
        students = []
        for i in range(self.num_students):
            profile = sample_profile(self.rng)
            student_rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
            student = SimulatedStudent(
                student_id=f"sim_{i:03d}",
                profile=profile,
                base_params=base,
                rng=student_rng,
            )
            for topic in self.kg.topics():
                student.init_topic(topic.id)
            students.append(student)
        return students

    def run(self, *, sessions_per_week: int = 5) -> SimulationResult:
        """Trivial runner: each week, each student does ``sessions_per_week``
        QUIZ_EXISTING actions over a uniformly random topic. This is a
        baseline used by tests; the real Q-learning training loop replaces
        the policy."""
        students = self.make_students()
        topics = [t.id for t in self.kg.topics()]
        logs = [StudentLog(student_id=s.id, profile=s.profile.name) for s in students]

        for _week in range(self.num_weeks):
            for student, log in zip(students, logs, strict=True):
                for _ in range(sessions_per_week):
                    topic = topics[int(student._rng.integers(0, len(topics)))]
                    student.respond(topic)
                    log.actions.append(topic)
                # Apply weekly forgetting.
                student.apply_decay(days_elapsed=7.0, half_life_days=self.half_life)
                log.true_mastery_history.append(student.all_true_mastery())
                log.belief_mastery_history.append(student.predictor.all_mastery())

        return SimulationResult(logs=logs, seed=self.seed)
