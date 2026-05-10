"""Evaluate trained KT models on the held-out split.

Computes AUC for each model and writes
``evaluation/kt_benchmark/results/eval_seed{seed}.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import pickle

from sklearn.metrics import roc_auc_score

from evaluation.kt_benchmark.data_loader import chronological_split, load_assistments
from src.config import seed_everything

DEFAULT_DATA = Path("data/assistments_2009/skill_builder_data_corrected_collapsed.csv")
RESULTS_DIR = Path("evaluation/kt_benchmark/results")


def evaluate(model, test_sequences) -> float:
    y_true: list[int] = []
    y_score: list[float] = []
    for seq in test_sequences:
        preds = model.predict_sequence(seq)
        for it, p in zip(seq.interactions, preds, strict=False):
            y_true.append(int(it.correct))
            y_score.append(float(p))
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--skip-dkt", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    sequences = load_assistments(args.data)
    _, test = chronological_split(sequences, train_frac=0.7)
    print(f"  {len(test)} test sequences")

    results: dict[str, float] = {}
    for name in ("per_skill_mean", "bkt", "dkt"):
        if name == "dkt" and args.skip_dkt:
            continue
        path = RESULTS_DIR / f"{name}_seed{args.seed}.pkl"
        if not path.exists():
            print(f"skipping {name}: {path} missing — run train_models.py first")
            continue
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        auc = evaluate(model, test)
        results[name] = auc
        print(f"  {name}: AUC = {auc:.4f}")

    out = RESULTS_DIR / f"eval_seed{args.seed}.json"
    out.write_text(json.dumps({"seed": args.seed, "auc": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
