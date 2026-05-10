from __future__ import annotations

import random
from datetime import date

import numpy as np
import pytest

from src.config import load_config, seed_everything
from src.scheduler.q_learning import (
    QLearningAgent,
    QLearningHyperparams,
    compute_reward,
)
from src.scheduler.replanner import replan
from src.scheduler.state import StateEncoder
from src.scheduler.thompson import ThompsonSamplingPolicy
from src.types import (
    ActionType,
    DisruptionType,
    DisruptionUpdate,
    SchedulerState,
    Session,
)

# ---------- state encoder ------------------------------------------


def make_encoder() -> StateEncoder:
    return StateEncoder.from_config(load_config())


def test_encoder_state_space_finite():
    enc = make_encoder()
    assert enc.num_states > 0
    assert enc.num_actions == 4


def test_encoder_round_trip_within_bins():
    enc = make_encoder()
    s = SchedulerState(
        fraction_mastered=0.5,
        days_remaining=20,
        num_at_risk=2,
        last_action=ActionType.QUIZ_EXISTING,
    )
    sid = enc.encode(s)
    assert 0 <= sid < enc.num_states
    decoded = enc.decode(sid)
    assert decoded["last_action"] == ActionType.QUIZ_EXISTING.value


def test_encoder_clamps_out_of_range():
    enc = make_encoder()
    s = SchedulerState(
        fraction_mastered=2.0,  # way over
        days_remaining=10_000,  # way over max
        num_at_risk=999,  # way over max
        last_action=ActionType.REST,
    )
    sid = enc.encode(s)
    assert 0 <= sid < enc.num_states


def test_encoder_distinguishes_different_states():
    enc = make_encoder()
    s1 = SchedulerState(0.1, 30, 0, ActionType.REST)
    s2 = SchedulerState(0.9, 5, 5, ActionType.INTRODUCE_NEW)
    assert enc.encode(s1) != enc.encode(s2)


# ---------- Q-learning agent ---------------------------------------


def make_agent() -> QLearningAgent:
    cfg = load_config()
    return QLearningAgent(
        encoder=StateEncoder.from_config(cfg),
        hyperparams=QLearningHyperparams.from_config(cfg),
    )


def test_agent_initial_q_table_is_zero():
    agent = make_agent()
    assert np.allclose(agent.q_table, 0.0)


def test_agent_select_action_returns_eligible():
    seed_everything(7)
    agent = make_agent()
    s = SchedulerState(0.4, 30, 1, ActionType.REST)
    eligible = [ActionType.INTRODUCE_NEW, ActionType.REST]
    chosen = agent.select_action(s, eligible)
    assert chosen in eligible


def test_agent_select_action_empty_eligible_raises():
    agent = make_agent()
    s = SchedulerState(0.4, 30, 1, ActionType.REST)
    with pytest.raises(ValueError):
        agent.select_action(s, [])


def test_agent_update_propagates_reward():
    agent = make_agent()
    agent.epsilon = 0.0  # greedy
    s = SchedulerState(0.4, 30, 1, ActionType.REST)
    s_next = SchedulerState(0.5, 29, 1, ActionType.INTRODUCE_NEW)
    agent.update(
        state=s,
        action=ActionType.INTRODUCE_NEW,
        reward=1.0,
        next_state=s_next,
        next_eligible=[ActionType.REST],
        done=False,
    )
    sid = agent.encoder.encode(s)
    aid = agent.encoder.action_index(ActionType.INTRODUCE_NEW)
    assert agent.q_table[sid, aid] > 0.0


def test_agent_greedy_after_training():
    """Train on a single (state, INTRODUCE_NEW) pair and verify the
    greedy policy picks INTRODUCE_NEW from the eligible set."""
    agent = make_agent()
    agent.epsilon = 0.0
    s = SchedulerState(0.4, 30, 1, ActionType.REST)
    s_next = SchedulerState(0.5, 29, 1, ActionType.INTRODUCE_NEW)
    eligible = [ActionType.INTRODUCE_NEW, ActionType.REST, ActionType.QUIZ_EXISTING]
    for _ in range(50):
        agent.update(s, ActionType.INTRODUCE_NEW, 1.0, s_next, [ActionType.REST], done=True)
        agent.update(s, ActionType.REST, -0.5, s_next, [ActionType.REST], done=True)
    chosen = agent.select_action(s, eligible, explore=False)
    assert chosen == ActionType.INTRODUCE_NEW


def test_agent_decay_epsilon():
    agent = make_agent()
    initial = agent.epsilon
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon < initial
    assert agent.epsilon >= agent.hyperparams.epsilon_end


def test_agent_save_and_load(tmp_path):
    agent = make_agent()
    agent.q_table[5, 2] = 0.42
    agent.epsilon = 0.123
    path = tmp_path / "qtable"
    agent.save(path)

    other = make_agent()
    other.load(path)
    assert other.q_table[5, 2] == pytest.approx(0.42)
    assert other.epsilon == pytest.approx(0.123)


# ---------- reward shaping -----------------------------------------


def test_reward_rewards_mastery_gain():
    cfg = load_config()
    base = compute_reward(
        mastery_gain=0.0, on_track=False, deadline_missed=False, num_at_risk=0, config=cfg
    )
    higher = compute_reward(
        mastery_gain=0.5, on_track=False, deadline_missed=False, num_at_risk=0, config=cfg
    )
    assert higher > base


def test_reward_penalises_deadline_miss():
    cfg = load_config()
    base = compute_reward(
        mastery_gain=0.0, on_track=False, deadline_missed=False, num_at_risk=0, config=cfg
    )
    miss = compute_reward(
        mastery_gain=0.0, on_track=False, deadline_missed=True, num_at_risk=0, config=cfg
    )
    assert miss < base


# ---------- Thompson Sampling --------------------------------------


def test_thompson_selects_from_eligible():
    rng = np.random.default_rng(7)
    policy = ThompsonSamplingPolicy(rng=rng)
    eligible = [ActionType.INTRODUCE_NEW, ActionType.REST]
    for _ in range(20):
        chosen = policy.select_action(eligible)
        assert chosen in eligible


def test_thompson_learns_to_prefer_rewarding_action():
    rng = np.random.default_rng(7)
    random.seed(7)
    policy = ThompsonSamplingPolicy(rng=rng)
    eligible = [ActionType.INTRODUCE_NEW, ActionType.REST]
    # Reward INTRODUCE_NEW, punish REST.
    for _ in range(200):
        a = policy.select_action(eligible)
        if a == ActionType.INTRODUCE_NEW:
            policy.update(a, reward=1.0)
        else:
            policy.update(a, reward=-1.0)
    expectations = policy.expectations()
    assert expectations[ActionType.INTRODUCE_NEW] > expectations[ActionType.REST]


def test_thompson_empty_eligible_raises():
    policy = ThompsonSamplingPolicy()
    with pytest.raises(ValueError):
        policy.select_action([])


# ---------- replanner ----------------------------------------------


def session(d: date, topic: str = "x", minutes: int = 30) -> Session:
    return Session(
        topic_id=topic, action=ActionType.QUIZ_EXISTING, scheduled_date=d, duration_minutes=minutes
    )


def test_replan_sick_day_pushes_sessions():
    cfg = load_config()
    today = date(2026, 5, 11)
    schedule = [
        session(today, "a"),
        session(today, "b"),
        session(date(2026, 5, 12), "c"),
    ]
    update = DisruptionUpdate(type=DisruptionType.SICK_DAY, payload={"date": today.isoformat()})
    new_schedule = replan(schedule, update, today=today, config=cfg)
    # No sessions remain on the sick day.
    assert all(s.scheduled_date != today for s in new_schedule)
    # The two pushed topics still appear somewhere.
    topics = {s.topic_id for s in new_schedule}
    assert topics == {"a", "b", "c"}


def test_replan_completed_externally_drops_topics():
    cfg = load_config()
    today = date(2026, 5, 11)
    schedule = [session(today, "a"), session(today, "b")]
    update = DisruptionUpdate(
        type=DisruptionType.COMPLETED_EXTERNALLY, payload={"topic_ids": ["a"]}
    )
    new_schedule = replan(schedule, update, today=today, config=cfg)
    assert {s.topic_id for s in new_schedule} == {"b"}


def test_replan_deadline_change_drops_past_deadline():
    cfg = load_config()
    today = date(2026, 5, 11)
    schedule = [
        session(date(2026, 5, 12), "a"),
        session(date(2026, 6, 1), "b"),  # past new deadline
    ]
    update = DisruptionUpdate(
        type=DisruptionType.DEADLINE_CHANGE,
        payload={"new_deadline": "2026-05-20"},
    )
    new_schedule = replan(schedule, update, today=today, config=cfg)
    # 'b' is repacked into the available window before the new deadline
    # (or dropped if no slot exists). Either way, no session is past it.
    assert all(s.scheduled_date <= date(2026, 5, 20) for s in new_schedule)
