# Evaluation Results

> All results below were produced from a single run with `seed=42`,
> against `config/default.yaml` as committed.
> Raw JSON / pickled artefacts live in each pipeline's `results/` directory.

---

## 1. KT Benchmark — `evaluation/kt_benchmark/`

**Dataset:** ASSISTments 2009 (`skill_builder_data_corrected_collapsed.csv`)
4,163 student sequences, 283,105 interactions, 149 skills.
Chronological 70/30 per-student split → 4,027 train / 4,027 test sequences.

**Result (`results/eval_seed42.json`):**

| Model | AUC |
|---|---:|
| `per_skill_mean` (baseline) | 0.6006 |
| **`bkt` (deployed predictor)** | **0.6859** |
| `dkt` (PyTorch LSTM, 3 epochs) | 0.6815 |

**Interpretation:** BKT outperforms the per-skill-mean baseline by 8.5
AUC points and is competitive with DKT despite being far simpler and
fully interpretable. This supports the design decision to deploy BKT
rather than DKT in the planner.

**Trained artefacts in `results/`:**

| File | Size | Notes |
|---|---:|---|
| `bkt_seed42.pkl` | 8.4 KB | per-skill BKT params |
| `dkt_seed42.pkl` | 248 KB | LSTM weights |
| `per_skill_mean_seed42.pkl` | 2.3 KB | per-skill correctness rates |
| `train_meta_seed42.json` | 130 B | training metadata |
| `eval_seed42.json` | 134 B | AUC numbers |

**Reproduce:**
```bash
python evaluation/kt_benchmark/train_models.py --seed 42
python evaluation/kt_benchmark/evaluate.py    --seed 42
```

---

## 2. Scheduler Comparison — `evaluation/scheduler_comparison/`

**Setup:** 6-topic linear DAG, 30 simulated students, 8 weeks, 5
sessions per week, mixed slow/average/fast profiles.
Q-learning agent trained for 5,000 episodes (full config budget).

**Result (`results/comparison_seed42.json`)** — coverage = fraction of
syllabus introduced by deadline; mastery = mean true mastery at
deadline:

| Policy | Coverage | Mean mastery |
|---|---:|---:|
| `rule-only` | ~0.483 | **~0.185** |
| `thompson` (Beta per action) | ~0.294 | ~0.144 |
| `q-learning` (5k episodes) | ~0.444 | ~0.165 |

**Interpretation:**

* The lifecycle-aware rule layer no longer treats every cold-start BKT
  prior as an already-introduced topic. This lowers coverage compared
  with the earlier permissive run but avoids unrealistic review/quiz
  actions on unseen topics.
* Rule-only has the highest mean mastery in this run because it spends
  less time exploring and concentrates practice on feasible topics.
* Thompson Sampling under-covers because its Beta posteriors don't
  encode prerequisite-aware preferences — it explores the action space
  broadly and spends effort on lower-value actions.
* Absolute mastery values are intentionally modest given the
  aggressive 7-day half-life decay and small session budget — see
  `docs/design_decisions.md` §10 on the sim-to-real gap.

**Trained artefacts in `results/`:**

| File | Size | Notes |
|---|---:|---|
| `qtable_seed42.npy` | 12.9 KB | 400×4 Q-table |
| `qtable_seed42.json` | 62 B | epsilon, dimensions |
| `comparison_seed42.json` | 1.4 KB | per-policy / per-profile metrics |

**Reproduce:**
```bash
python evaluation/scheduler_comparison/run_comparison.py --seed 42
```

---

## 3. LLM Component Evals — `evaluation/llm_components/`

**Provider:** `AnthropicProvider` with `claude-sonnet-4-20250514`
(model + temperature pinned in `config/default.yaml`).

**Result (`results/llm_eval_seed42.json`):**

### KG extraction

| Metric | Value |
|---|---:|
| topic precision | 0.42 |
| **topic recall** | **1.00** |
| edge precision | 0.00 |
| edge recall | 0.00 |
| predicted topic count | 12 |
| predicted edge count | 11 |

The LLM is more granular than the 5-topic ground truth (extracts 12
topics covering all 5 gold topics plus extras like "data preprocessing"
and "feature engineering"). All 5 gold topics are recovered (recall
1.00). Edge precision is 0 because the extra topics shift the edge
endpoints; the right metric for this case is "edges *projected onto
the gold topic set*" — flagged for the report's discussion §10.

### Quiz generation (5 trials)

| Metric | Value |
|---|---:|
| successful trials | **5/5** |
| total questions generated | 10 |
| **citation rate** | **1.00** |
| MCQ share | 0.50 |

Every generated question carried at least one `[chunk_id]` citation —
the faithfulness invariant holds under real generation.

### Grader agreement (3 fixtures)

| Metric | Value |
|---|---:|
| **agreement rate** | **1.00** |

The LLM grader matched the reference rubric on all three test cases
(correct, incorrect, partial).

### Explanation faithfulness

| Metric | Value |
|---|---:|
| citation count | 4 |
| valid citations | 4 |
| **valid-citation rate** | **1.00** |

Every citation in the explanation pointed to a real chunk ID from the
provided context.

**Reproduce:**
```bash
python evaluation/llm_components/run_llm_evals.py --seed 42 --provider anthropic
```

---

## 4. End-to-end smoke test — `scripts/smoke_test.py`

Not a benchmark; a single deterministic pass through all four
orchestrator flows against the real Anthropic API. Used to catch
integration regressions that mock-based tests miss.

| Stage | Wall-clock | Verified |
|---|---:|---|
| `setup_flow` | 4.4 s | KG is a valid DAG; 5 topics + 4 edges; BKT priors in [0, 1] |
| `weekly_flow` | 13.4 s | At least one chunk ingested; non-trivial summary |
| `session_flow` | 15.0 s | Mastery moved 0.05 → 0.72; explanation contains `[chunk_id]` tags |
| `disruption_flow` | 1.6 s | Free text parsed into typed `DisruptionUpdate` |
| **Total** | **34.4 s** | All 12 invariants pass |

**Reproduce:**
```bash
python scripts/smoke_test.py
```

---

## What is captured vs. what would be follow-up

Captured in this run:

* All seven prompt templates exercised at least once against the real LLM.
* Three policies compared head-to-head with a properly trained Q-table.
* All trained model weights and Q-tables saved on disk.
* Hard invariants (DAG, mastery bounds, citation tags, provider abstraction, seeded RNG) verified at runtime.

Not yet captured (planned follow-up):

* Disruption recovery time (only coverage and mastery are reported in the scheduler comparison).
* LLM quiz difficulty calibrated against simulated learners at low/medium/high mastery.
* Cached LLM responses for offline / CI runs.
