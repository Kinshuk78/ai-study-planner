"""Train BKT, DKT, and per-skill mean on ASSISTments 2009.

Saves trained artefacts under ``evaluation/kt_benchmark/results/``.

Usage
-----

    python evaluation/kt_benchmark/train_models.py --seed 42

The dataset must be downloaded first; see
``data/assistments_2009/README.md``.
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

from evaluation.kt_benchmark.data_loader import chronological_split, load_assistments
from evaluation.kt_benchmark.models import BKTBenchmarkModel, DKTModel, PerSkillMean
from src.config import seed_everything

DEFAULT_DATA = Path("data/assistments_2009/skill_builder_data_corrected_collapsed.csv")
RESULTS_DIR = Path("evaluation/kt_benchmark/results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--skip-dkt", action="store_true", help="Skip DKT training (slow).")
    args = parser.parse_args()

    seed_everything(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    sequences = load_assistments(args.data)
    print(f"  {len(sequences)} student sequences")

    train, _ = chronological_split(sequences, train_frac=0.7)
    print(f"  {len(train)} training sequences")

    print("Fitting per-skill mean ...")
    psm = PerSkillMean()
    psm.fit(train)
    with open(RESULTS_DIR / f"per_skill_mean_seed{args.seed}.pkl", "wb") as fh:
        pickle.dump(psm, fh)

    print("Fitting BKT ...")
    bkt = BKTBenchmarkModel()
    bkt.fit(train)
    with open(RESULTS_DIR / f"bkt_seed{args.seed}.pkl", "wb") as fh:
        pickle.dump(bkt, fh)

    if not args.skip_dkt:
        print("Fitting DKT ...")
        dkt = DKTModel(hidden_size=64, num_epochs=3)
        dkt.fit(train)
        with open(RESULTS_DIR / f"dkt_seed{args.seed}.pkl", "wb") as fh:
            pickle.dump(dkt, fh)

    meta = {
        "seed": args.seed,
        "num_train_sequences": len(train),
        "data_path": str(args.data),
    }
    (RESULTS_DIR / f"train_meta_seed{args.seed}.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
