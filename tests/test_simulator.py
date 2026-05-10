from __future__ import annotations

import numpy as np
import pytest

from src.bkt import default_params
from src.kg import KnowledgeGraph, Topic
from src.simulator import (
    SimulatedStudent,
    SimulationRunner,
    get_profile,
)


def make_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for tid in ("a", "b"):
        kg.add_topic(Topic(id=tid, name=tid))
    kg.add_prerequisite("a", "b")
    return kg


# ---------- profiles -----------------------------------------------


def test_profiles_load():
    fast = get_profile("fast")
    slow = get_profile("slow")
    assert fast.transit_multiplier > slow.transit_multiplier


# ---------- student model ------------------------------------------


def test_student_p_correct_in_unit_interval():
    profile = get_profile("average")
    student = SimulatedStudent(
        student_id="s",
        profile=profile,
        base_params=default_params(),
        rng=np.random.default_rng(0),
    )
    student.init_topic("a")
    p = student.p_correct("a")
    assert 0.0 <= p <= 1.0


def test_student_response_updates_belief():
    profile = get_profile("average")
    student = SimulatedStudent(
        student_id="s",
        profile=profile,
        base_params=default_params(),
        rng=np.random.default_rng(0),
    )
    student.init_topic("a")
    before = student.predictor.mastery("a")
    student.respond("a")
    after = student.predictor.mastery("a")
    # Belief always changes, in either direction.
    assert before != after


def test_student_decay_reduces_true_mastery():
    profile = get_profile("fast")
    student = SimulatedStudent(
        student_id="s",
        profile=profile,
        base_params=default_params(),
        rng=np.random.default_rng(0),
    )
    student.init_topic("a")
    # Force true mastery up.
    student._true_mastery["a"] = 0.9
    student.apply_decay(days_elapsed=7, half_life_days=7)
    assert student.true_mastery("a") == pytest.approx(0.45, abs=0.05)


# ---------- runner / reproducibility -------------------------------


def test_runner_is_seeded_and_deterministic():
    kg = make_kg()
    r1 = SimulationRunner(kg, seed=99).run(sessions_per_week=2)
    r2 = SimulationRunner(kg, seed=99).run(sessions_per_week=2)
    assert r1.seed == r2.seed
    assert [log.actions for log in r1.logs] == [log.actions for log in r2.logs]


def test_runner_logs_per_student_per_week():
    kg = make_kg()
    runner = SimulationRunner(kg, seed=7)
    result = runner.run(sessions_per_week=2)
    assert len(result.logs) == runner.num_students
    for log in result.logs:
        assert len(log.true_mastery_history) == runner.num_weeks
        assert len(log.belief_mastery_history) == runner.num_weeks


def test_fast_profile_outpaces_slow_on_average():
    """Fast students should reach higher mean mastery than slow students."""
    kg = make_kg()
    runner = SimulationRunner(kg, seed=42)
    students = runner.make_students()
    # Quiz everyone many times on topic 'a'.
    for s in students:
        for _ in range(50):
            s.respond("a")
    fast = [s.true_mastery("a") for s in students if s.profile.name == "fast"]
    slow = [s.true_mastery("a") for s in students if s.profile.name == "slow"]
    if fast and slow:
        assert sum(fast) / len(fast) >= sum(slow) / len(slow)
