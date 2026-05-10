# Personalised AI Study Planner

A study planner that ingests a course syllabus, models how a learner's
topic mastery evolves, plans study across weeks with daily granularity,
adapts to disruptions, and explains its decisions using lecturer-provided
materials. The system was built to demonstrate the integration of six
distinct AI paradigms in a single coherent application.

The empirical scope is deliberately bounded: scheduling is evaluated on
*simulated learners* and the knowledge-tracing benchmark uses
ASSISTments 2009. The honest framing — that the system adapts correctly
to *modelled* mechanisms, not validated *real* ones — is preserved
throughout.

---

## The six paradigms

| # | Paradigm | Role in the system |
|---|----------|--------------------|
| 1 | **Knowledge graph** | A directed acyclic graph of topics with `is_prerequisite_of` edges. Constructed by LLM extraction and human verification. Backed by NetworkX. |
| 2 | **Bayesian Knowledge Tracing (BKT)** | Per-topic mastery estimates with an exponential forgetting-curve decay. Deployed as the predictor — chosen over DKT for interpretability and cold-start behaviour. |
| 3 | **Tabular Q-learning** | Learns a policy over four action types: `INTRODUCE_NEW`, `REVIEW_WEAKEST`, `QUIZ_EXISTING`, `REST`. State space ≈ 400 cells; trained in simulation. |
| 4 | **Rule-based logic** | Hard feasibility layer: prerequisites, daily/weekly capacity, deadline reachability, disruption reflow. Outputs the *eligible action set*; Q-learning chooses among that set. Rules are not relaxed by the policy. |
| 5 | **LLM + RAG** | Handles every natural-language interface (KG extraction, quiz generation, grading, self-report parsing, explanation, weekly summary). Two retrieval strategies: focused (KG-scoped similarity) and graph-traversal (walks prerequisite edges). |
| 6 | **Spaced repetition / forgetting model** | Exponential decay applied between sessions. Review behaviour *emerges* from decaying mastery raising the at-risk count, which the scheduler then prioritises — there is no hard-coded "schedule reviews" rule. |

---

## How it works

The system orchestrates four deterministic flows:

```
            ┌─────────────────────────────────────────────────────────┐
syllabus ──►│ 1. Setup    LLM extracts KG → human verification → BKT │
            │             priors → week-1 plan                        │
            └─────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────▼─────────────────────────────────┐
materials ─►│ 2. Weekly   ingest materials (chunk + embed + link to KG) │
            │             → advance scheduler → graph-traversal RAG     │
            │             → LLM weekly summary                          │
            └───────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────▼──────────────────────────────────┐
            │ 3. Session  focused RAG → LLM quiz → student responses     │
            │             → grading → BKT update → Q-learning state      │
            │             update → graph-traversal RAG → LLM explanation │
            └────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────▼─────────────────────────────────┐
report ────►│ 4. Disruption  LLM parses to structured update            │
            │                (sick_day / deadline_change / capacity     │
            │                change / completed_externally) → reflow    │
            └───────────────────────────────────────────────────────────┘
```

The decomposition between the rule layer (hard constraints) and the
Q-learning policy (soft preferences) is invariant: rules are enforced;
preference is learned. They are never collapsed.

---

## Quickstart

### Install

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you intend to use the real LLM backend, copy your API key into a
`.env` file at the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

### Run the demo

```bash
streamlit run src/ui/streamlit_app.py
```

The default LLM provider is `mock`, which returns canned responses so
the demo runs without API access. Switch to `anthropic` from the
sidebar once your key is configured.

### Run the test suite

```bash
pytest                                       # all tests
pytest --cov=src --cov-report=term-missing   # with coverage
```

### Run the simulated trace

```bash
python scripts/sim_demo.py                     # 3 profiles × 4 weeks
python scripts/sim_demo.py --weeks 8 --sessions-per-week 6
```

Walks one simulated student per profile through several weeks of study
sessions and prints what the rule layer chose, what the student
answered, how BKT belief updated, and how forgetting decay shifted
end-of-week mastery.

### End-to-end smoke test against the real LLM

```bash
python scripts/smoke_test.py
```

Exercises all four orchestrator flows against the live API in roughly
30 seconds. Used to catch integration regressions that mock-based tests
cannot see.

---

## Evaluation

Three reproducible pipelines, each writing JSON results to
`evaluation/{component}/results/`. See [`evaluation/RESULTS.md`](evaluation/RESULTS.md)
for full numbers and discussion.

### Knowledge tracing benchmark

Compares BKT against a per-skill mean baseline and DKT (PyTorch LSTM)
on ASSISTments 2009.

```bash
python evaluation/kt_benchmark/train_models.py --seed 42
python evaluation/kt_benchmark/evaluate.py    --seed 42
```

| Model | AUC |
|---|---:|
| per-skill mean (baseline) | 0.6006 |
| **BKT (deployed)** | **0.6859** |
| DKT (PyTorch LSTM) | 0.6815 |

Place the dataset at
`data/assistments_2009/skill_builder_data_corrected_collapsed.csv`
(see the README in that directory).

### Scheduler comparison

Rule-only vs. Thompson Sampling vs. tabular Q-learning across three
learner profiles (slow, average, fast).

```bash
python evaluation/scheduler_comparison/run_comparison.py --seed 42
```

| Policy | Coverage | Mean mastery |
|---|---:|---:|
| rule-only | **1.00** | ~0.110 |
| Thompson Sampling | ~0.80 | ~0.033 |
| Q-learning | **1.00** | ~0.085 |

### LLM component evaluations

Per-task evaluations: knowledge-graph extraction precision/recall,
quiz citation rate, grader agreement, explanation faithfulness.

```bash
python evaluation/llm_components/run_llm_evals.py --seed 42 --provider anthropic
```

---

## Configuration

All tunable parameters live in [`config/default.yaml`](config/default.yaml) —
BKT priors and decay half-life, Q-learning hyperparameters and state-bin
counts, scheduler capacity limits, RAG chunk size and retrieval depth,
LLM provider/model/temperature, and the random seed. Modules read these
at runtime; nothing is hard-coded.

LLM prompt templates live in [`config/prompts.yaml`](config/prompts.yaml).
There are no prompts written inline in Python.

---

## Repository layout

```
.
├── config/                    tunable parameters and prompt templates
├── src/
│   ├── kg/                    knowledge graph schema, store, traversal
│   ├── bkt/                   BKT predictor + forgetting decay
│   ├── scheduler/             rule layer + Q-learning + Thompson + replanner
│   ├── llm/                   provider abstraction, task modules, JSONL cache
│   ├── rag/                   chunker, embedder, vector store, focused/traversal RAG
│   ├── simulator/             simulated student model, profiles, runner
│   ├── orchestrator/          the four flows (setup / weekly / session / disruption)
│   └── ui/                    Streamlit demo
├── tests/                     unit + integration tests
├── evaluation/                three reproducible benchmarks + summary
├── scripts/                   sim_demo.py, smoke_test.py
└── data/                      datasets and example syllabus (mostly gitignored)
```

---

## Tech stack

- **Python 3.10+**
- **NumPy / Pandas / NetworkX** for numerical work and graphs
- **PyTorch** for the DKT comparison model only — not the deployed predictor
- **Anthropic** and **Ollama** SDKs for LLM access
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings
- **ChromaDB** + a default in-memory vector store
- **Streamlit** for the demo UI
- **pytest** for testing, **ruff** for lint/format, **mypy** for type checking

All dependencies are pinned in `requirements.txt`.

---

## Reproducibility

- A single `random_seed` in `config/default.yaml` propagates to NumPy,
  PyTorch, the Python `random` module, and the simulator.
- Every evaluation script accepts `--seed` and writes results to
  `evaluation/{component}/results/{name}_seed{seed}.json`.
- Trained Q-tables and KT model artefacts are committed alongside the
  code so reviewers can reproduce evaluation numbers without retraining.
- The LLM model version is pinned in `config/default.yaml`. Responses
  for the canonical example syllabus are cached in
  `data/example_syllabus/llm_cache.jsonl`, so the demo and CI can run
  offline.

---

## Hard invariants

These are correctness conditions for the empirical claims and are
enforced in code:

- The scheduler never places a topic before its prerequisites reach
  `mastery_threshold`.
- The Q-learning state is the discretised abstraction
  `(fraction_mastered, days_remaining, num_at_risk, last_action)` —
  never a raw mastery vector.
- The knowledge graph is validated as a DAG after every construction.
- BKT mastery is always in `[0, 1]`.
- LLM explanations always carry `[chunk_id]` citation tags; the
  generation step raises if a citation is missing.
- Code outside `src/llm/` never imports `anthropic` or `ollama`
  directly — every call goes through the `LLMProvider` abstraction.

---

## Honest limitations

The report is graded on its critical reflection, so the following are
acknowledged rather than papered over:

- Scheduling evaluation is on simulated learners. The simulator is a
  *model*, not the world; results do not generalise to real learners
  without further validation.
- The KT benchmark uses a 70/30 chronological split per student. AUC
  numbers reflect ASSISTments 2009 skill-level prediction and should
  not be read as evidence about the planner's effectiveness for any
  individual learner.
- LLM evaluation sample sizes are small (a single canonical syllabus,
  three grader fixtures, five quiz trials). The metrics are descriptive,
  not statistically significant.
- The forgetting model uses an Ebbinghaus-style exponential decay with
  a default 7-day half-life. The half-life is not calibrated against
  any individual learner's behaviour.

These are discussed in greater depth in the project report.

---

## Out of scope

The following are explicitly not implemented and not planned:

- Real-user deployment, ethics clearance, participant recruitment
- Cross-subject scheduling
- Multiple deadline types (only a single course-end deadline is supported)
- Audio/video transcription (PDF and plain text only)
- Long-form essay grading (MCQ and short-answer only)
- Group/collaborative study features
- Mobile UI, accessibility, internationalisation
- Production authentication and persistent multi-user storage
