"""Streamlit demo wiring all four orchestrator flows.

Run with: ``streamlit run src/ui/streamlit_app.py``

The UI is intentionally minimal — its only job is to surface the four
orchestrator flows. Heavy lifting stays in :mod:`src.orchestrator`.
"""

from __future__ import annotations

# --- self-bootstrap so `streamlit run src/ui/streamlit_app.py` works
# without requiring PYTHONPATH=. to be set externally.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import date

import streamlit as st

from src.config import load_config, seed_everything
from src.llm.provider import LLMProvider, get_provider
from src.orchestrator import handle_disruption, run_session, run_setup, run_weekly
from src.rag import InMemoryVectorStore
from src.types import QuizQuestion


def _provider() -> LLMProvider:
    if "provider" not in st.session_state:
        st.session_state.provider = get_provider(st.session_state.get("provider_name"))
    provider: LLMProvider = st.session_state.provider
    return provider


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["random_seed"])
    st.set_page_config(page_title="AI Study Planner", layout="wide")
    st.title("Personalised AI Study Planner")

    with st.sidebar:
        st.header("Settings")
        st.session_state.provider_name = st.selectbox(
            "LLM provider",
            options=["mock", "anthropic", "ollama"],
            index=["mock", "anthropic", "ollama"].index(cfg["llm"]["provider"])
            if cfg["llm"]["provider"] in ("mock", "anthropic", "ollama")
            else 0,
        )
        st.write(f"Seed: {cfg['random_seed']}")

    tab_setup, tab_weekly, tab_session, tab_disruption = st.tabs(
        ["Setup", "Weekly", "Session", "Disruption"]
    )

    if "store" not in st.session_state:
        st.session_state.store = InMemoryVectorStore()
    if "schedule" not in st.session_state:
        st.session_state.schedule = []

    # ----- Setup tab -----
    with tab_setup:
        st.subheader("Upload syllabus and verify the knowledge graph")
        syllabus_text = st.text_area("Syllabus text", height=200)
        if st.button("Extract KG and initialise", disabled=not syllabus_text):
            result = run_setup(
                syllabus_text=syllabus_text,
                provider=_provider(),
                config=cfg,
                today=date.today(),
            )
            st.session_state.kg = result.kg
            st.session_state.predictor = result.predictor
            st.session_state.schedule = result.initial_plan
            st.success(
                f"Extracted {len(result.kg.topics())} topics, "
                f"{len(result.kg.edges())} edges, planned "
                f"{len(result.initial_plan)} sessions."
            )
        if "kg" in st.session_state:
            st.write("Topics:", [t.id for t in st.session_state.kg.topics()])
            st.write("Edges:", [(e.source, e.target) for e in st.session_state.kg.edges()])

    # ----- Weekly tab -----
    with tab_weekly:
        st.subheader("Ingest weekly materials and generate a summary")
        if "kg" not in st.session_state:
            st.info("Run Setup first.")
        else:
            week_no = st.number_input("Week number", min_value=1, value=1, step=1)
            topic_options = [t.id for t in st.session_state.kg.topics()]
            sel = st.selectbox("Topic for these materials", topic_options)
            body = st.text_area("Material text", height=160)
            if st.button("Ingest week"):
                weekly = run_weekly(
                    week_number=int(week_no),
                    materials=[(sel, body)],
                    kg=st.session_state.kg,
                    predictor=st.session_state.predictor,
                    store=st.session_state.store,
                    provider=_provider(),
                    mastery_diff={},
                    sessions_log="(weekly UI demo)",
                )
                st.success(f"Ingested {weekly.chunks_added} chunks.")
                st.text_area("Weekly summary", value=weekly.summary, height=180)

    # ----- Session tab -----
    with tab_session:
        st.subheader("Run a study session")
        if "kg" not in st.session_state or len(st.session_state.store) == 0:
            st.info("Run Setup and ingest a week of materials first.")
        else:
            topic_options = [t.id for t in st.session_state.kg.topics()]
            sel = st.selectbox("Topic", topic_options, key="session_topic")
            if st.button("Run session"):
                # Auto-answer with the reference answer (UI demo). The
                # session flow itself is the same component the
                # simulator drives.
                def answer_fn(q: QuizQuestion) -> str:
                    return q.answer

                session = run_session(
                    topic_id=sel,
                    kg=st.session_state.kg,
                    predictor=st.session_state.predictor,
                    store=st.session_state.store,
                    provider=_provider(),
                    answer_fn=answer_fn,
                    config=cfg,
                )
                st.write("Quiz results:")
                for g in session.graded:
                    st.write(
                        f"- [{g.response.question.type.value}] "
                        f"score={g.score:.2f} feedback={g.feedback}"
                    )
                st.text_area("Explanation", value=session.explanation, height=180)
                st.write("Suggested next action:", session.next_action.value)

    # ----- Disruption tab -----
    with tab_disruption:
        st.subheader("Report a disruption")
        if "kg" not in st.session_state:
            st.info("Run Setup first.")
        else:
            text = st.text_area(
                "What changed?",
                placeholder="e.g. I was sick today / deadline moved to ...",
                height=120,
            )
            if st.button("Apply disruption", disabled=not text):
                disruption = handle_disruption(
                    report_text=text,
                    schedule=st.session_state.schedule,
                    provider=_provider(),
                    config=cfg,
                )
                st.session_state.schedule = disruption.new_schedule
                st.write(
                    "Parsed update:",
                    disruption.update.type.value,
                    disruption.update.payload,
                )
                st.success(disruption.confirmation)


if __name__ == "__main__":
    main()
