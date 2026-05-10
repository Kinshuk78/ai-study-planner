"""BKT parameter dataclass and config-driven defaults."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import load_config


@dataclass(frozen=True)
class BKTParams:
    """Standard BKT parameters. All values must be in [0, 1]."""

    L0: float  # P(initial mastery)
    slip: float  # P(incorrect | mastered)
    guess: float  # P(correct | not mastered)
    transit: float  # P(transition to mastered after a learning opportunity)

    def __post_init__(self) -> None:
        for name, value in (
            ("L0", self.L0),
            ("slip", self.slip),
            ("guess", self.guess),
            ("transit", self.transit),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"BKT parameter '{name}' must be in [0, 1], got {value}")


def default_params() -> BKTParams:
    """BKT priors loaded from ``config/default.yaml``."""
    cfg = load_config()["bkt"]["priors"]
    return BKTParams(
        L0=cfg["L0"],
        slip=cfg["slip"],
        guess=cfg["guess"],
        transit=cfg["transit"],
    )
