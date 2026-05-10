"""Config loader. All modules read tunable parameters via :func:`load_config`."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path) as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


@lru_cache(maxsize=4)
def load_prompts(path: str | Path = DEFAULT_PROMPTS_PATH) -> dict[str, dict[str, str]]:
    with open(path) as fh:
        data: dict[str, dict[str, str]] = yaml.safe_load(fh)
    return data


def seed_everything(seed: int | None = None) -> int:
    """Seed numpy, torch, and python random from the config seed.

    Returns the seed actually used. If ``seed`` is ``None`` the value is
    read from ``config/default.yaml``.
    """
    if seed is None:
        seed = int(load_config()["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
