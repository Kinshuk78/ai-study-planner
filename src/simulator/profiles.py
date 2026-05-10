"""Simulated learner profiles.

Each profile modulates BKT priors via multipliers, producing a
:class:`BKTParams` for the simulated student. Profile shares are sampled
from a single seeded RNG so populations are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bkt.params import BKTParams
from src.config import load_config


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    transit_multiplier: float
    slip_multiplier: float
    population_share: float


def get_profile(name: str) -> ProfileSpec:
    cfg = load_config()["simulator"]["profiles"][name]
    return ProfileSpec(
        name=name,
        transit_multiplier=cfg["transit_multiplier"],
        slip_multiplier=cfg["slip_multiplier"],
        population_share=cfg["population_share"],
    )


def all_profiles() -> list[ProfileSpec]:
    cfg = load_config()["simulator"]["profiles"]
    return [get_profile(name) for name in cfg]


def sample_profile(rng: np.random.Generator) -> ProfileSpec:
    profiles = all_profiles()
    weights = np.array([p.population_share for p in profiles], dtype=float)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(profiles), p=weights))
    return profiles[idx]


def apply_profile(base: BKTParams, profile: ProfileSpec) -> BKTParams:
    return BKTParams(
        L0=base.L0,
        slip=_clip(base.slip * profile.slip_multiplier),
        guess=base.guess,
        transit=_clip(base.transit * profile.transit_multiplier),
    )


def _clip(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
