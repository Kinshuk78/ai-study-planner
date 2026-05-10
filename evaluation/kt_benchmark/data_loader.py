"""ASSISTments 2009 loader.

The dataset is gitignored. Download instructions live in
``data/assistments_2009/README.md``.

Returns sequences keyed by ``user_id`` so each model can be evaluated
on per-student response sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Interaction:
    skill_id: str
    correct: int  # 0 or 1


@dataclass
class StudentSequence:
    student_id: str
    interactions: list[Interaction]


def load_assistments(path: str | Path) -> list[StudentSequence]:
    """Load the canonical ASSISTments 2009 skill-builder CSV.

    Expects columns at minimum: ``user_id``, ``skill_id`` (or ``skill_name``),
    ``correct``. Drops rows missing any of those.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ASSISTments file not found at {path}. "
            "See data/assistments_2009/README.md for download instructions."
        )
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    skill_col = "skill_id" if "skill_id" in df.columns else "skill_name"
    df = df.dropna(subset=["user_id", skill_col, "correct"])
    df["user_id"] = df["user_id"].astype(str)
    df[skill_col] = df[skill_col].astype(str)
    df["correct"] = df["correct"].astype(int)

    sequences: list[StudentSequence] = []
    for uid, group in df.groupby("user_id"):
        interactions = [
            Interaction(skill_id=row[skill_col], correct=int(row["correct"]))
            for _, row in group.iterrows()
        ]
        sequences.append(StudentSequence(student_id=uid, interactions=interactions))
    return sequences


def chronological_split(
    sequences: list[StudentSequence], train_frac: float = 0.7
) -> tuple[list[StudentSequence], list[StudentSequence]]:
    """Per-student chronological split: first ``train_frac`` of each sequence
    is train, the rest is test."""
    train, test = [], []
    for seq in sequences:
        n_train = int(len(seq.interactions) * train_frac)
        if n_train < 1 or n_train >= len(seq.interactions):
            continue
        train.append(StudentSequence(seq.student_id, seq.interactions[:n_train]))
        test.append(StudentSequence(seq.student_id, seq.interactions[n_train:]))
    return train, test
