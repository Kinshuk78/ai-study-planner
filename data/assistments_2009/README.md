# ASSISTments 2009 — download instructions

The KT benchmark uses the canonical *skill_builder_data.csv* file from
the public ASSISTments 2009 release. The dataset is gitignored.

## Download

1. Visit the ASSISTments dataset page:
   https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data
2. Download the **skill_builder_data_corrected_collapsed.csv** file
   (the corrected/collapsed release is preferred; it deduplicates the
   raw skill_builder_data.csv).
3. Place the CSV at
   ``data/assistments_2009/skill_builder_data_corrected_collapsed.csv``.

Sanity check: roughly 4,163 students, 283,105 interactions, 149 unique
skills.

## Expected schema

The loader expects (at minimum) these columns:

* ``user_id`` — student identifier
* ``skill_id`` *or* ``skill_name`` — skill / topic identifier
* ``correct`` — 0 or 1

Additional columns are ignored.

## Reproducing the benchmark

```bash
python evaluation/kt_benchmark/train_models.py --seed 42
python evaluation/kt_benchmark/evaluate.py    --seed 42
```

Results are written to ``evaluation/kt_benchmark/results/``.
