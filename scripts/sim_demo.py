"""Run a simulated test trace.

Builds a tiny knowledge graph, instantiates one simulated student per
profile (slow / average / fast), and traces them through several weeks
of study sessions. The rule layer picks the action; the simulated
student answers; BKT belief updates; forgetting decay is applied at
the end of each week.

Run::

    python scripts/sim_demo.py
    python scripts/sim_demo.py --weeks 6 --sessions-per-week 4

This is a *trace*, not a benchmark. For statistics, use
``evaluation/scheduler_comparison/run_comparison.py``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from itertools import pairwise
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.bkt import default_params
from src.config import load_config, seed_everything
from src.kg import KnowledgeGraph, Topic
from src.scheduler.rules import eligible_actions, select_topic_for_action
from src.simulator.profiles import all_profiles, apply_profile
from src.simulator.student_model import SimulatedStudent
from src.types import ActionType


def make_demo_kg() -> KnowledgeGraph:
    """A 4-topic linear DAG: vectors -> matrices -> regression -> classification."""
    kg = KnowledgeGraph()
    chain = [
        ("vectors", "Vectors"),
        ("matrices", "Matrices"),
        ("regression", "Regression"),
        ("classification", "Classification"),
    ]
    for tid, name in chain:
        kg.add_topic(Topic(id=tid, name=name))
    for (a, _), (b, _) in pairwise(chain):
        kg.add_prerequisite(a, b)
    return kg


def fmt_mastery_row(label: str, beliefs: dict[str, float], topic_order: list[str]) -> str:
    cells = [f"{beliefs.get(t, 0.0):.2f}" for t in topic_order]
    return f"  {label:<22} {' | '.join(cells)}"


def run_student(
    *,
    profile_name: str,
    kg: KnowledgeGraph,
    weeks: int,
    sessions_per_week: int,
    seed: int,
    cfg: dict,
) -> None:
    profile = next(p for p in all_profiles() if p.name == profile_name)
    rng = np.random.default_rng(seed)
    base = default_params()
    student = SimulatedStudent(
        student_id=f"demo_{profile.name}",
        profile=profile,
        base_params=base,
        rng=rng,
    )
    student.params = apply_profile(base, profile)
    student.predictor.params = student.params
    for t in kg.topics():
        student.init_topic(t.id)

    topic_order = [t.id for t in kg.topics()]
    print(f"\n{'=' * 78}")
    print(f"  Profile: {profile.name.upper():<8}  "
          f"transit_mult={profile.transit_multiplier}  "
          f"slip_mult={profile.slip_multiplier}")
    print('=' * 78)
    print(f"  topics:                {' | '.join(topic_order)}")
    print(fmt_mastery_row("week 0 (priors):", student.predictor.all_mastery(), topic_order))

    half_life = cfg["bkt"]["decay"]["half_life_days"]

    for week in range(1, weeks + 1):
        print(f"\n  -- week {week} --")
        for s in range(1, sessions_per_week + 1):
            beliefs = student.predictor.all_mastery()
            eligible = eligible_actions(
                candidate_topic_ids=topic_order,
                bkt_estimates=beliefs,
                kg=kg,
                schedule=[],
                today=date.today(),
                config=cfg,
            )
            # Greedy rule policy: first non-REST action that is eligible.
            action = next((a for a in eligible if a != ActionType.REST), ActionType.REST)
            topic_id = select_topic_for_action(
                action,
                candidate_topic_ids=topic_order,
                bkt_estimates=beliefs,
                kg=kg,
                schedule=[],
                config=cfg,
            )
            if topic_id is None:
                print(f"    s{s}: {action.value:<16} (no eligible topic) — REST")
                continue
            before = beliefs[topic_id]
            correct = student.respond(topic_id)
            after = student.predictor.mastery(topic_id)
            tick = "✓" if correct else "✗"
            print(
                f"    s{s}: {action.value:<16} topic={topic_id:<14} "
                f"answered {tick}   mastery {before:.2f} → {after:.2f}"
            )

        # End of week: apply forgetting decay.
        student.apply_decay(7.0, half_life)
        belief = student.predictor.all_mastery()
        true = student.all_true_mastery()
        print(fmt_mastery_row(f"end-of-week-{week} belief:", belief, topic_order))
        print(fmt_mastery_row("end-of-week true mastery:", true, topic_order))

    # Summary.
    final_belief = student.predictor.all_mastery()
    final_true = student.all_true_mastery()
    mastered = sum(1 for m in final_belief.values() if m >= cfg["bkt"]["mastery_threshold"])
    print(
        f"\n  summary: {mastered}/{len(topic_order)} topics believed mastered  "
        f"(true mean = {sum(final_true.values()) / len(final_true):.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--sessions-per-week", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    seed_everything(args.seed)
    kg = make_demo_kg()

    print(f"Knowledge graph (linear DAG): {' -> '.join(t.id for t in kg.topics())}")
    print(f"Weeks: {args.weeks}   sessions/week: {args.sessions_per_week}   seed: {args.seed}")
    print(f"Mastery threshold: {cfg['bkt']['mastery_threshold']}   "
          f"At-risk threshold: {cfg['bkt']['at_risk_threshold']}   "
          f"Forgetting half-life: {cfg['bkt']['decay']['half_life_days']} days")

    for i, profile_name in enumerate(["slow", "average", "fast"]):
        run_student(
            profile_name=profile_name,
            kg=kg,
            weeks=args.weeks,
            sessions_per_week=args.sessions_per_week,
            seed=args.seed + i,  # different seed per student
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
