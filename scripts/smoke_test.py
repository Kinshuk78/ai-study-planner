"""End-to-end smoke test against the real Anthropic API.

Runs all four orchestrator flows on a tiny syllabus and verifies the
hard invariants. Used to catch integration regressions that mock-based
tests can't see (prompt/response shape mismatches, JSON-mode behaviour,
citation formatting, etc.).

Run from the project root::

    python scripts/smoke_test.py

The script auto-loads ``ANTHROPIC_API_KEY`` from ``.env`` if present.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't require python-dotenv."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(_PROJECT_ROOT / ".env")

from src.config import load_config, seed_everything
from src.llm.cache import CachedProvider
from src.llm.provider import get_provider
from src.llm.tasks.explainer import extract_citations
from src.orchestrator import (
    handle_disruption,
    run_session,
    run_setup,
    run_weekly,
)
from src.rag import InMemoryVectorStore
from src.types import ActionType, DisruptionType

# A tiny but realistic syllabus — keeps token use low.
SYLLABUS = (
    "Introduction to Machine Learning. Topics covered, in order:\n"
    "1. Linear algebra (vectors, matrices, dot products).\n"
    "2. Probability (distributions, expectation, variance).\n"
    "3. Linear regression (requires linear algebra).\n"
    "4. Logistic regression (requires linear regression and probability).\n"
    "5. Model evaluation (requires logistic regression)."
)

WEEKLY_MATERIAL = {
    "linear_algebra": (
        "A vector is an ordered tuple of numbers. A matrix is a "
        "rectangular grid of numbers; matrix-vector multiplication "
        "computes a dot product per row. The dot product measures "
        "alignment between two vectors."
    ),
    "linear_regression": (
        "Linear regression fits a line y = w*x + b by minimising the "
        "sum of squared residuals between predicted and observed values. "
        "It can be solved in closed form using the normal equations or "
        "iteratively via gradient descent."
    ),
}


def banner(stage: str) -> None:
    print(f"\n{'=' * 60}\n {stage}\n{'=' * 60}")


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ✓ {message}")
    else:
        print(f"  ✗ {message}")
        raise AssertionError(message)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to .env or export it.")
        return 1

    cfg = load_config()
    seed_everything(cfg["random_seed"])
    print(f"Model: {cfg['llm']['anthropic_model']}")
    print(f"Seed:  {cfg['random_seed']}")

    provider = get_provider("anthropic")
    cache_status = (
        f"cached → {provider.cache_path}" if isinstance(provider, CachedProvider) else "uncached"
    )
    print(f"Provider: {provider.__class__.__name__} ({cache_status})")
    timings: dict[str, float] = {}

    # -----------------------------------------------------------------
    # Setup flow
    # -----------------------------------------------------------------
    banner("1/4  setup_flow")
    t0 = time.time()
    setup = run_setup(
        syllabus_text=SYLLABUS,
        provider=provider,
        config=cfg,
        today=date.today(),
    )
    timings["setup"] = time.time() - t0

    setup.kg.validate_dag()
    topic_ids = {t.id for t in setup.kg.topics()}
    print(f"  topics:  {sorted(topic_ids)}")
    print(f"  edges:   {[(e.source, e.target) for e in setup.kg.edges()]}")
    print(f"  plan:    {len(setup.initial_plan)} sessions")
    check(len(topic_ids) >= 3, "extracted >= 3 topics")
    check(len(setup.kg.edges()) >= 1, "extracted >= 1 prerequisite edge")
    check(
        all(0.0 <= setup.predictor.mastery(t) <= 1.0 for t in topic_ids),
        "BKT priors initialised in [0, 1]",
    )

    # Pick a topic that exists in the LLM-extracted KG for material ingestion.
    candidate_material_topic = next(
        (tid for tid in topic_ids if any(key in tid for key in ("linear_algebra", "regression"))),
        sorted(topic_ids)[0],
    )

    # -----------------------------------------------------------------
    # Weekly flow — ingest one piece of material
    # -----------------------------------------------------------------
    banner("2/4  weekly_flow")
    store = InMemoryVectorStore()
    t0 = time.time()
    weekly = run_weekly(
        week_number=1,
        materials=[
            (
                candidate_material_topic,
                WEEKLY_MATERIAL.get(candidate_material_topic, WEEKLY_MATERIAL["linear_algebra"]),
            ),
        ],
        kg=setup.kg,
        predictor=setup.predictor,
        store=store,
        provider=provider,
        mastery_diff={candidate_material_topic: (0.10, 0.10)},
        sessions_log="(smoke test)",
    )
    timings["weekly"] = time.time() - t0
    print(f"  chunks added: {weekly.chunks_added}")
    print(f"  summary preview: {weekly.summary[:200]}{'…' if len(weekly.summary) > 200 else ''}")
    check(weekly.chunks_added > 0, "at least one chunk ingested")
    check(len(weekly.summary) > 50, "summary is non-trivially long")

    # -----------------------------------------------------------------
    # Session flow — run one quiz + explanation
    # -----------------------------------------------------------------
    banner("3/4  session_flow")

    def perfect_answer(q):
        # Simulate a strong learner — return the reference answer verbatim.
        return q.answer

    before_mastery = setup.predictor.mastery(candidate_material_topic)
    t0 = time.time()
    session = run_session(
        topic_id=candidate_material_topic,
        kg=setup.kg,
        predictor=setup.predictor,
        store=store,
        provider=provider,
        answer_fn=perfect_answer,
        config=cfg,
    )
    timings["session"] = time.time() - t0
    after_mastery = setup.predictor.mastery(candidate_material_topic)

    print(f"  graded {len(session.graded)} responses")
    for g in session.graded:
        print(
            f"    [{g.response.question.type.value}] score={g.score:.2f} feedback={g.feedback[:60]}"
        )
    print(
        f"  explanation preview: {session.explanation[:200]}{'…' if len(session.explanation) > 200 else ''}"
    )
    print(f"  mastery: {before_mastery:.3f} -> {after_mastery:.3f}")
    print(f"  next action: {session.next_action.value}")

    citations = extract_citations(session.explanation)
    check(len(session.graded) >= 1, "session produced at least one graded response")
    check(after_mastery >= before_mastery - 1e-6, "mastery did not decrease on perfect answers")
    check(len(citations) >= 1, "explanation contains [chunk_id] citations")
    check(session.next_action in set(ActionType), "next action is a valid ActionType")

    # -----------------------------------------------------------------
    # Disruption flow
    # -----------------------------------------------------------------
    banner("4/4  disruption_flow")
    t0 = time.time()
    disruption = handle_disruption(
        report_text="I was sick today and couldn't study.",
        schedule=setup.initial_plan,
        provider=provider,
        config=cfg,
        today=date.today(),
    )
    timings["disruption"] = time.time() - t0
    print(f"  parsed type:  {disruption.update.type.value}")
    print(f"  payload:      {disruption.update.payload}")
    print(f"  confirmation: {disruption.confirmation}")
    print(f"  schedule len: {len(disruption.new_schedule)}")
    check(
        isinstance(disruption.update.type, DisruptionType),
        "disruption parsed into a typed DisruptionUpdate",
    )
    check(disruption.confirmation, "confirmation message produced")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    banner("RESULT")
    total = sum(timings.values())
    for stage, t in timings.items():
        print(f"  {stage:<10}  {t:6.2f}s")
    print(f"  {'TOTAL':<10}  {total:6.2f}s")
    if isinstance(provider, CachedProvider):
        s = provider.stats()
        print(f"  cache:     {s['hits']} hits / {s['misses']} misses (size={s['size']})")
    print("\nAll smoke checks passed ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
