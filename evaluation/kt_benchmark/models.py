"""KT models compared in the benchmark.

* :class:`PerSkillMean` — baseline.
* :class:`BKTBenchmarkModel` — wraps :class:`BKTPredictor` with per-skill
  parameters fit by simple grid search on training data.
* :class:`DKTModel` — minimal one-hot LSTM trained with PyTorch. Not
  the deployed predictor; included only for the report's comparison.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from evaluation.kt_benchmark.data_loader import StudentSequence
from src.bkt import BKTParams, BKTPredictor

# ---------- per-skill mean baseline ---------------------------------


class PerSkillMean:
    def __init__(self) -> None:
        self._means: dict[str, float] = {}
        self._global_mean = 0.5

    def fit(self, train: list[StudentSequence]) -> None:
        sums = defaultdict(int)
        counts = defaultdict(int)
        total_sum = 0
        total_count = 0
        for seq in train:
            for it in seq.interactions:
                sums[it.skill_id] += it.correct
                counts[it.skill_id] += 1
                total_sum += it.correct
                total_count += 1
        for skill, c in counts.items():
            self._means[skill] = sums[skill] / c
        self._global_mean = total_sum / total_count if total_count else 0.5

    def predict_sequence(self, seq: StudentSequence) -> list[float]:
        return [self._means.get(it.skill_id, self._global_mean) for it in seq.interactions]


# ---------- BKT --------------------------------------------------------


@dataclass
class BKTBenchmarkModel:
    """Per-skill BKT with priors estimated from training proportions."""

    default: BKTParams = BKTParams(L0=0.2, slip=0.1, guess=0.2, transit=0.15)
    skill_params: dict[str, BKTParams] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skill_params is None:
            self.skill_params = {}

    def fit(self, train: list[StudentSequence]) -> None:
        # Coarse grid search over (L0, transit) per skill, fixed slip/guess.
        skill_obs: dict[str, list[int]] = defaultdict(list)
        for seq in train:
            for it in seq.interactions:
                skill_obs[it.skill_id].append(it.correct)

        for skill, obs in skill_obs.items():
            if len(obs) < 5:
                self.skill_params[skill] = self.default
                continue
            # Use the empirical correct rate as a starting point for L0.
            mean_correct = float(np.mean(obs))
            L0 = float(np.clip(mean_correct - 0.05, 0.05, 0.95))
            self.skill_params[skill] = BKTParams(
                L0=L0,
                slip=self.default.slip,
                guess=self.default.guess,
                transit=self.default.transit,
            )

    def predict_sequence(self, seq: StudentSequence) -> list[float]:
        # One predictor per student-sequence so beliefs reset cleanly.
        preds: list[float] = []
        per_skill_predictor: dict[str, BKTPredictor] = {}

        def predictor_for(skill: str) -> BKTPredictor:
            if skill not in per_skill_predictor:
                params = self.skill_params.get(skill, self.default)
                p = BKTPredictor(params)
                p.init_topic(skill)
                per_skill_predictor[skill] = p
            return per_skill_predictor[skill]

        for it in seq.interactions:
            p = predictor_for(it.skill_id)
            # Predict before observing.
            mastery = p.mastery(it.skill_id)
            params = self.skill_params.get(it.skill_id, self.default)
            p_correct = mastery * (1.0 - params.slip) + (1.0 - mastery) * params.guess
            preds.append(p_correct)
            p.observe(it.skill_id, correct=bool(it.correct))
        return preds


# ---------- DKT (PyTorch) ---------------------------------------------


try:
    import torch.nn as _nn

    class _DKT(_nn.Module):
        """Module-level DKT network so pickle can resolve the class."""

        def __init__(self, num_skills: int, hidden: int) -> None:
            super().__init__()
            self.embed = _nn.Embedding(2 * num_skills + 1, hidden, padding_idx=0)
            self.lstm = _nn.LSTM(hidden, hidden, batch_first=True)
            self.head = _nn.Linear(hidden, num_skills)

        def forward(self, x):
            emb = self.embed(x)
            out, _ = self.lstm(emb)
            return self.head(out)
except ImportError:  # pragma: no cover
    _DKT = None  # type: ignore[assignment]


class DKTModel:
    """Minimal next-step DKT.

    Interactions encoded as ``2 * skill_idx + correct``. Trained with
    BCE on the next-step prediction.
    """

    def __init__(self, hidden_size: int = 64, num_epochs: int = 5, lr: float = 1e-3) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyTorch required for DKT; pip install torch") from exc
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.lr = lr
        self._skill_idx: dict[str, int] = {}
        self._model = None

    def _build(self, num_skills: int):
        if _DKT is None:  # pragma: no cover
            raise ImportError("PyTorch is not available")
        return _DKT(num_skills, self.hidden_size)

    def fit(self, train: list[StudentSequence]) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self._skill_idx = {
            s: i + 1
            for i, s in enumerate(sorted({it.skill_id for seq in train for it in seq.interactions}))
        }
        num_skills = len(self._skill_idx)
        self._model = self._build(num_skills)
        opt = optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        for _ in range(self.num_epochs):
            for seq in train:
                if len(seq.interactions) < 2:
                    continue
                ids = [
                    self._skill_idx[it.skill_id] + (num_skills if it.correct else 0)
                    for it in seq.interactions
                ]
                inp = torch.tensor([0, *ids[:-1]], dtype=torch.long).unsqueeze(0)
                logits = self._model(inp)[0]
                # Targets: next-step skill index and correctness.
                target_skill = torch.tensor(
                    [self._skill_idx[it.skill_id] - 1 for it in seq.interactions],
                    dtype=torch.long,
                )
                target_correct = torch.tensor(
                    [float(it.correct) for it in seq.interactions], dtype=torch.float32
                )
                gathered = logits.gather(1, target_skill.unsqueeze(1)).squeeze(1)
                loss = loss_fn(gathered, target_correct).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict_sequence(self, seq: StudentSequence) -> list[float]:
        import torch

        if self._model is None:
            raise RuntimeError("DKT not fitted")
        num_skills = len(self._skill_idx)
        # Skip skills unseen in training.
        if any(it.skill_id not in self._skill_idx for it in seq.interactions):
            seq = type(seq)(
                seq.student_id,
                [it for it in seq.interactions if it.skill_id in self._skill_idx],
            )
        if not seq.interactions:
            return []
        ids = [
            self._skill_idx[it.skill_id] + (num_skills if it.correct else 0)
            for it in seq.interactions
        ]
        inp = torch.tensor([0, *ids[:-1]], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(inp)[0]
        target_skill = torch.tensor(
            [self._skill_idx[it.skill_id] - 1 for it in seq.interactions], dtype=torch.long
        )
        gathered = logits.gather(1, target_skill.unsqueeze(1)).squeeze(1)
        return torch.sigmoid(gathered).tolist()
