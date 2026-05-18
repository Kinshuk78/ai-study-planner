"""Compare rule-only, Thompson Sampling, and Q-learning policies.

For each policy and each profile (slow / average / fast), runs N
simulated students for M weeks, then reports:

  - syllabus coverage at deadline (fraction of topics ever introduced)
  - mean true mastery at deadline
  - mean disruption recovery time (n/a in this minimal harness)

Writes results to
``evaluation/scheduler_comparison/results/comparison_seed{seed}.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json

import numpy as np

from src.bkt import default_params
from src.config import load_config, seed_everything
from src.kg import KnowledgeGraph, Topic
from src.scheduler.q_learning import (
    QLearningAgent,
    QLearningHyperparams,
    compute_reward,
)
from src.scheduler.rules import eligible_actions, select_topic_for_action
from src.scheduler.state import StateEncoder
from src.scheduler.thompson import ThompsonSamplingPolicy
from src.simulator.profiles import all_profiles, apply_profile
from src.simulator.student_model import SimulatedStudent
from src.types import ActionType, SchedulerState

RESULTS_DIR = Path("evaluation/scheduler_comparison/results")


def make_demo_kg() -> KnowledgeGraph:
    """A 6-topic linear DAG used for the harness."""
    kg = KnowledgeGraph()
    chain = [
        ("intro", "Introduction"),
        ("linear_algebra", "Linear Algebra"),
        ("probability", "Probability"),
        ("regression", "Regression"),
        ("classification", "Classification"),
        ("evaluation", "Evaluation"),
    ]
    from itertools import pairwise

    for tid, name in chain:
        kg.add_topic(Topic(id=tid, name=name))
    for (a, _), (b, _) in pairwise(chain):
        kg.add_prerequisite(a, b)
    return kg


def build_state(
    student: SimulatedStudent, last_action: ActionType, days_remaining: int, config: dict
) -> SchedulerState:
    cfg = config["bkt"]
    threshold = cfg["mastery_threshold"]
    at_risk = cfg["at_risk_threshold"]
    beliefs = student.predictor.all_mastery()
    fraction_mastered = (
        sum(1 for v in beliefs.values() if v >= threshold) / len(beliefs) if beliefs else 0.0
    )
    num_at_risk = sum(1 for v in beliefs.values() if v < at_risk)
    return SchedulerState(
        fraction_mastered=fraction_mastered,
        days_remaining=days_remaining,
        num_at_risk=num_at_risk,
        last_action=last_action,
    )


def run_policy(
    name: str,
    *,
    kg: KnowledgeGraph,
    config: dict,
    seed: int,
    weeks: int,
    sessions_per_week: int,
    num_students: int,
    agent: QLearningAgent | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    base = default_params()
    profiles = all_profiles()

    coverage_per_profile: dict[str, list[float]] = {p.name: [] for p in profiles}
    mastery_per_profile: dict[str, list[float]] = {p.name: [] for p in profiles}

    for i in range(num_students):
        profile_idx = int(rng.integers(0, len(profiles)))
        profile = profiles[profile_idx]
        params = apply_profile(base, profile)
        student_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        student = SimulatedStudent(
            student_id=f"s_{i}",
            profile=profile,
            base_params=base,
            rng=student_rng,
        )
        student.params = params
        student.predictor.params = params
        for t in kg.topics():
            student.init_topic(t.id)

        ts_policy = ThompsonSamplingPolicy(rng=rng) if name == "thompson" else None
        last_action = ActionType.REST
        introduced: set[str] = set()

        total_days = weeks * 7
        for w in range(weeks):
            days_remaining = total_days - w * 7
            for _ in range(sessions_per_week):
                beliefs = student.predictor.all_mastery()
                eligible = eligible_actions(
                    candidate_topic_ids=list(beliefs.keys()),
                    bkt_estimates=beliefs,
                    kg=kg,
                    schedule=[],
                    today=__import__("datetime").date.today(),
                    config=config,
                )
                if name == "rule":
                    action = next((a for a in eligible if a != ActionType.REST), ActionType.REST)
                elif name == "thompson":
                    assert ts_policy is not None
                    action = ts_policy.select_action(eligible)
                else:  # qlearning
                    assert agent is not None
                    state = build_state(student, last_action, days_remaining, config)
                    action = agent.select_action(state, eligible, explore=False)

                topic_id = select_topic_for_action(
                    action,
                    candidate_topic_ids=list(beliefs.keys()),
                    bkt_estimates=beliefs,
                    kg=kg,
                    schedule=[],
                    config=config,
                )
                if topic_id is not None:
                    if action == ActionType.INTRODUCE_NEW:
                        introduced.add(topic_id)
                    correct = student.respond(topic_id)
                    if name == "thompson":
                        ts_policy.update(action, reward=1.0 if correct else -1.0)

                last_action = action

            student.apply_decay(7.0, config["bkt"]["decay"]["half_life_days"])

        coverage = len(introduced) / len(beliefs)
        mean_mastery = float(np.mean(list(student.all_true_mastery().values())))
        coverage_per_profile[profile.name].append(coverage)
        mastery_per_profile[profile.name].append(mean_mastery)

    summary = {
        "policy": name,
        "by_profile": {
            p.name: {
                "coverage_mean": float(np.mean(coverage_per_profile[p.name]))
                if coverage_per_profile[p.name]
                else 0.0,
                "mastery_mean": float(np.mean(mastery_per_profile[p.name]))
                if mastery_per_profile[p.name]
                else 0.0,
                "n": len(coverage_per_profile[p.name]),
            }
            for p in profiles
        },
    }
    return summary


def train_qlearning(*, kg: KnowledgeGraph, config: dict, seed: int) -> QLearningAgent:
    """Brief training loop used by the harness so the comparison can run
    end-to-end. The full training script lives elsewhere."""
    encoder = StateEncoder.from_config(config)
    agent = QLearningAgent(encoder=encoder, hyperparams=QLearningHyperparams.from_config(config))
    rng = np.random.default_rng(seed)
    base = default_params()
    profiles = all_profiles()

    weeks = config["simulator"]["num_weeks"]
    sessions_per_week = 5
    episodes = config["q_learning"]["num_training_episodes"]

    for _ep in range(episodes):
        profile = profiles[int(rng.integers(0, len(profiles)))]
        student_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        student = SimulatedStudent(
            student_id="train",
            profile=profile,
            base_params=base,
            rng=student_rng,
        )
        for t in kg.topics():
            student.init_topic(t.id)
        last_action = ActionType.REST
        for w in range(weeks):
            days_remaining = (weeks - w) * 7
            for _ in range(sessions_per_week):
                beliefs = student.predictor.all_mastery()
                eligible = eligible_actions(
                    candidate_topic_ids=list(beliefs.keys()),
                    bkt_estimates=beliefs,
                    kg=kg,
                    schedule=[],
                    today=__import__("datetime").date.today(),
                    config=config,
                )
                state = build_state(student, last_action, days_remaining, config)
                action = agent.select_action(state, eligible)
                topic_id = select_topic_for_action(
                    action,
                    candidate_topic_ids=list(beliefs.keys()),
                    bkt_estimates=beliefs,
                    kg=kg,
                    schedule=[],
                    config=config,
                )
                before = float(np.mean(list(beliefs.values())))
                if topic_id is not None:
                    student.respond(topic_id)
                after = float(np.mean(list(student.predictor.all_mastery().values())))
                num_at_risk = sum(
                    1
                    for v in student.predictor.all_mastery().values()
                    if v < config["bkt"]["at_risk_threshold"]
                )
                reward = compute_reward(
                    mastery_gain=after - before,
                    on_track=after >= 0.5,
                    deadline_missed=False,
                    num_at_risk=num_at_risk,
                    config=config,
                )
                next_state = build_state(student, action, max(1, days_remaining - 1), config)
                next_eligible = eligible_actions(
                    candidate_topic_ids=list(student.predictor.all_mastery().keys()),
                    bkt_estimates=student.predictor.all_mastery(),
                    kg=kg,
                    schedule=[],
                    today=__import__("datetime").date.today(),
                    config=config,
                )
                done = w == weeks - 1
                agent.update(state, action, reward, next_state, next_eligible, done=done)
                last_action = action
            student.apply_decay(7.0, config["bkt"]["decay"]["half_life_days"])
        agent.decay_epsilon()
    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-students", type=int, default=30)
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--sessions-per-week", type=int, default=5)
    args = parser.parse_args()

    seed_everything(args.seed)
    config = load_config()
    kg = make_demo_kg()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training Q-learning ...")
    agent = train_qlearning(kg=kg, config=config, seed=args.seed)
    agent.save(RESULTS_DIR / f"qtable_seed{args.seed}")

    summaries = []
    for name in ("rule", "thompson", "qlearning"):
        print(f"Running policy: {name}")
        summary = run_policy(
            name,
            kg=kg,
            config=config,
            seed=args.seed,
            weeks=args.weeks,
            sessions_per_week=args.sessions_per_week,
            num_students=args.num_students,
            agent=agent if name == "qlearning" else None,
        )
        summaries.append(summary)

    out = {"seed": args.seed, "policies": summaries}
    path = RESULTS_DIR / f"comparison_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
