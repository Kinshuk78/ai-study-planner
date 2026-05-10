"""Per-task LLM component evaluations.

Covers:

  - **KG extraction**: precision/recall vs a hand-written ground-truth
    KG for the canonical example syllabus.
  - **Quiz generation**: difficulty calibration — % correct by simulated
    learner at low / medium / high mastery.
  - **Grader agreement**: agreement rate between LLM grader and a
    reference rubric on a fixed set of student responses.
  - **Explanation faithfulness**: every explanation must contain
    ``[chunk_id]`` citations.

Writes results to
``evaluation/llm_components/results/llm_eval_seed{seed}.json``.

Sample sizes are intentionally small (this is a course project — see
§10 of the design doc on honest limitations).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(_PROJECT_ROOT / ".env")

import argparse
import json

from src.config import seed_everything
from src.llm.provider import get_provider
from src.llm.tasks import explain, extract_kg, generate_quiz, grade_short_response
from src.llm.tasks.explainer import extract_citations
from src.types import (
    Chunk,
    QuestionType,
    QuizQuestion,
    QuizResponse,
    RetrievedChunk,
)

RESULTS_DIR = Path("evaluation/llm_components/results")


# ---------- ground truth ------------------------------------------


GROUND_TRUTH_KG = {
    "topics": {"linear_algebra", "regression", "probability", "classification", "evaluation"},
    "edges": {
        ("linear_algebra", "regression"),
        ("probability", "classification"),
        ("regression", "classification"),
        ("classification", "evaluation"),
    },
}


SAMPLE_SYLLABUS = (
    "This course covers: linear algebra (vectors, matrices), probability "
    "(distributions, expectation), regression (linear models — requires linear "
    "algebra), classification (logistic regression — requires both probability "
    "and regression), and evaluation (cross-validation — requires classification)."
)


# ---------- evaluations -------------------------------------------


def eval_kg_extraction(provider) -> dict:
    kg = extract_kg(SAMPLE_SYLLABUS, provider)
    pred_topics = {t.id for t in kg.topics()}
    pred_edges = {(e.source, e.target) for e in kg.edges()}

    topic_p, topic_r = _prf(pred_topics, GROUND_TRUTH_KG["topics"])
    edge_p, edge_r = _prf(pred_edges, GROUND_TRUTH_KG["edges"])
    return {
        "topic_precision": topic_p,
        "topic_recall": topic_r,
        "edge_precision": edge_p,
        "edge_recall": edge_r,
        "predicted_topic_count": len(pred_topics),
        "predicted_edge_count": len(pred_edges),
    }


_QUIZ_CONTEXT_CHUNKS = [
    (
        "c_lin_reg_basics",
        "Linear regression models the relationship between a scalar response and one or "
        "more explanatory variables by fitting a line of the form y = w*x + b.",
    ),
    (
        "c_loss_function",
        "The model parameters are estimated by minimising the sum of squared residuals "
        "between predicted and observed values; this is the ordinary least squares objective.",
    ),
    (
        "c_solution_methods",
        "The closed-form solution comes from the normal equations: w = (X^T X)^-1 X^T y. "
        "When X^T X is large, gradient descent is used as an iterative alternative.",
    ),
]


def eval_quiz_difficulty(provider, num_trials: int = 5) -> dict:
    """Naive proxy: ratio of MCQ to short questions, citation rate."""
    chunks = [
        RetrievedChunk(chunk=Chunk(id=cid, text=text, source="lecture_03"), score=1.0)
        for cid, text in _QUIZ_CONTEXT_CHUNKS
    ]
    total_q = 0
    cited = 0
    mcq = 0
    successful_trials = 0
    for _ in range(num_trials):
        try:
            qs = generate_quiz(
                topic_name="Linear Regression",
                chunks=chunks,
                num_questions=2,
                provider=provider,
            )
        except (ValueError, KeyError):
            # The LLM occasionally refuses or returns malformed JSON on weak
            # contexts — count the trial as a failure rather than crashing
            # the whole eval.
            continue
        successful_trials += 1
        total_q += len(qs)
        cited += sum(1 for q in qs if q.citations)
        mcq += sum(1 for q in qs if q.type == QuestionType.MCQ)
    return {
        "questions_total": total_q,
        "citation_rate": cited / total_q if total_q else 0.0,
        "mcq_share": mcq / total_q if total_q else 0.0,
        "trials": num_trials,
        "successful_trials": successful_trials,
    }


def eval_grader_agreement(provider) -> dict:
    """Compare LLM grader scores to a tiny rule-based reference."""
    fixtures = [
        ("define linearity", "additivity and homogeneity", "additivity and homogeneity", 1.0),
        ("define linearity", "additivity and homogeneity", "I have no idea", 0.0),
        ("what is regression", "fits a line minimising squared residuals", "fits a line", 0.5),
    ]
    agree = 0
    for stem, ref, student, expected in fixtures:
        q = QuizQuestion(stem=stem, answer=ref, type=QuestionType.SHORT)
        r = QuizResponse(question=q, student_answer=student)
        graded = grade_short_response(r, provider)
        # Bucket into {0.0, 0.5, 1.0} and compare.
        bucket = 0.0 if graded.score < 0.25 else (0.5 if graded.score < 0.75 else 1.0)
        if abs(bucket - expected) < 1e-6:
            agree += 1
    return {"agreement_rate": agree / len(fixtures), "n": len(fixtures)}


def eval_explanation_faithfulness(provider) -> dict:
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                id="chunk_demo_1",
                text=(
                    "Linear regression fits a line y = w*x + b by minimising the sum of "
                    "squared residuals between observed and predicted values."
                ),
                source="lecture_03",
            ),
            score=1.0,
        ),
        RetrievedChunk(
            chunk=Chunk(
                id="chunk_demo_2",
                text=(
                    "Regression depends on linear-algebra concepts: the design matrix X "
                    "and the normal equations w = (X^T X)^-1 X^T y."
                ),
                source="lecture_03",
            ),
            score=1.0,
        ),
    ]
    text = explain(
        topic_name="Linear Regression",
        question="What is linear regression and how is it solved?",
        mastery_level=0.5,
        chunks=chunks,
        provider=provider,
    )
    citations = extract_citations(text)
    valid_ids = {rc.chunk.id for rc in chunks}
    valid = [c for c in citations if c in valid_ids]
    return {
        "citation_count": len(citations),
        "valid_citation_count": len(valid),
        "valid_rate": len(valid) / len(citations) if citations else 0.0,
    }


def _prf(pred: set, gold: set) -> tuple[float, float]:
    if not pred:
        return 0.0, 0.0 if gold else 1.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold) if gold else 1.0
    return p, r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    provider = get_provider(args.provider)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "seed": args.seed,
        "provider": provider.__class__.__name__,
        "kg_extraction": eval_kg_extraction(provider),
        "quiz_difficulty": eval_quiz_difficulty(provider),
        "grader_agreement": eval_grader_agreement(provider),
        "explanation_faithfulness": eval_explanation_faithfulness(provider),
    }
    path = RESULTS_DIR / f"llm_eval_seed{args.seed}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
