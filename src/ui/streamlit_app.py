"""Streamlit demo wiring all four orchestrator flows.

Run with: ``streamlit run src/ui/streamlit_app.py``
"""

from __future__ import annotations

# --- self-bootstrap so `streamlit run src/ui/streamlit_app.py` works
# without requiring PYTHONPATH=. to be set externally.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEMO_DIR = _PROJECT_ROOT / "data" / "demo_academic"

from datetime import date

import pandas as pd
import streamlit as st

from src.config import load_config, seed_everything
from src.llm.provider import LLMProvider, get_provider
from src.orchestrator import handle_disruption, run_session, run_setup, run_weekly
from src.rag import InMemoryVectorStore
from src.scheduler.rules import select_topic_for_action
from src.types import ActionType, QuizQuestion, Session

_WORKFLOW = (
    ("Setup", "Knowledge graph and BKT priors"),
    ("Weekly", "Materials, embeddings, summary"),
    ("Session", "Quiz, grading, mastery update"),
    ("Disruption", "Parse update and reflow plan"),
)

_TABLE_HEIGHT_SMALL = 176
_TABLE_HEIGHT_MEDIUM = 240


def _read_demo_file(name: str) -> str:
    path = _DEMO_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _demo_material_for(topic_id: str) -> str:
    files = {
        "linear_algebra": "linear_algebra_material.txt",
        "regression": "regression_material.txt",
    }
    return _read_demo_file(files.get(topic_id, "linear_algebra_material.txt"))


def _provider() -> LLMProvider:
    provider_name = st.session_state.get("provider_name", "mock")
    if (
        "provider" not in st.session_state
        or st.session_state.get("provider_instance_name") != provider_name
    ):
        st.session_state.provider = get_provider(provider_name)
        st.session_state.provider_instance_name = provider_name
    provider: LLMProvider = st.session_state.provider
    return provider


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #0b1020;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: #070b14;
            border-right: 1px solid #1f2937;
        }
        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 16, 32, 0.94);
            border-bottom: 1px solid #1f2937;
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2.25rem;
            padding-bottom: 4rem;
        }
        .stApp {
            color: #e5e7eb;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: #f8fafc;
        }
        p, li, label, span {
            color: #d1d5db;
        }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #253044;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stMetric"] label {
            color: #9ca3af;
            font-size: 0.82rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 1.65rem;
        }
        .hero {
            border: 1px solid #253044;
            border-radius: 8px;
            background: linear-gradient(135deg, #111827 0%, #172033 58%, #0f172a 100%);
            padding: 24px 28px;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
            margin-bottom: 18px;
        }
        .eyebrow {
            color: #38bdf8;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .hero-title {
            color: #f8fafc;
            font-size: 2.15rem;
            line-height: 1.1;
            font-weight: 760;
            margin: 0 0 8px;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1rem;
            max-width: 980px;
            margin: 0;
        }
        .panel {
            border: 1px solid #253044;
            border-radius: 8px;
            background: #111827;
            padding: 18px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
            margin-bottom: 14px;
        }
        .panel-title {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 720;
            margin-bottom: 4px;
        }
        .panel-caption {
            color: #9ca3af;
            font-size: 0.88rem;
            margin-bottom: 12px;
        }
        .decision {
            border: 1px solid #1d4ed8;
            border-radius: 8px;
            background: linear-gradient(135deg, #0f1f3d 0%, #111827 72%);
            padding: 18px;
            min-height: 154px;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
        }
        .decision-label {
            color: #38bdf8;
            font-size: 0.75rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .decision-title {
            color: #f8fafc;
            font-size: 1.48rem;
            font-weight: 760;
            line-height: 1.16;
            margin-bottom: 8px;
        }
        .decision-copy {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin: 0;
        }
        .quality-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }
        .quality-item {
            border: 1px solid #253044;
            border-radius: 8px;
            background: #0f172a;
            padding: 12px;
        }
        .quality-state {
            color: #67e8f9;
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .quality-name {
            color: #f8fafc;
            font-weight: 700;
            margin-top: 3px;
        }
        .quality-detail {
            color: #9ca3af;
            font-size: 0.82rem;
            margin-top: 3px;
        }
        .workflow {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 16px;
        }
        .workflow-step {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
            padding: 12px 14px;
        }
        .workflow-step.done {
            border-color: #86efac;
            background: #f0fdf4;
        }
        .workflow-step.active {
            border-color: #93c5fd;
            background: #eff6ff;
        }
        .workflow-index {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .workflow-name {
            color: #111827;
            font-size: 0.98rem;
            font-weight: 730;
            margin-top: 4px;
        }
        .workflow-desc {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 2px;
        }
        div[data-testid="stTabs"] button {
            font-weight: 650;
            color: #cbd5e1;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #38bdf8;
        }
        textarea, input, div[data-baseweb="select"] {
            border-radius: 8px !important;
        }
        textarea, input {
            background: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
        }
        div[data-baseweb="select"] > div {
            background: #0f172a !important;
            border-color: #334155 !important;
        }
        div[data-baseweb="select"] span {
            color: #f8fafc !important;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 650;
            border: 1px solid #334155;
            background: #1f2937;
            color: #f8fafc;
        }
        .stButton > button[kind="primary"] {
            background: #2563eb;
            border-color: #3b82f6;
            color: #ffffff;
        }
        .stButton > button:hover {
            border-color: #38bdf8;
            color: #ffffff;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #253044;
            background: #111827;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #253044;
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stAlert"] {
            background: #102033;
            color: #dbeafe;
            border: 1px solid #1d4ed8;
            border-radius: 8px;
        }
        div[data-testid="stCaptionContainer"] p {
            color: #9ca3af;
        }
        code {
            background: #1f2937 !important;
            color: #e5e7eb !important;
        }
        @media (max-width: 900px) {
            .workflow {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 1.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _panel(title: str, caption: str = "") -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


def _render_header(cfg: dict) -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Enterprise learning operations dashboard</div>
          <div class="hero-title">Personalised AI Study Planner</div>
          <p class="hero-subtitle">
            Deterministic planning, interpretable mastery tracking, RAG-grounded study
            sessions, and disruption recovery for academic schedules. Current seed:
            <strong>{cfg["random_seed"]}</strong>.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _status_counts() -> dict[str, float | int | str]:
    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    topics = len(kg.topics()) if kg is not None else 0
    edges = len(kg.edges()) if kg is not None else 0
    chunks = len(st.session_state.get("store", []))
    sessions = len(st.session_state.get("schedule", []))
    if predictor is not None and predictor.all_mastery():
        values = list(predictor.all_mastery().values())
        mean_mastery = sum(values) / len(values)
        mastery_label = f"{mean_mastery:.0%}"
    else:
        mastery_label = "Not started"
    return {
        "topics": topics,
        "edges": edges,
        "chunks": chunks,
        "sessions": sessions,
        "mastery": mastery_label,
    }


def _render_status_strip() -> None:
    counts = _status_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Topics", counts["topics"])
    c2.metric("Prerequisite edges", counts["edges"])
    c3.metric("Material chunks", counts["chunks"])
    c4.metric("Planned sessions", counts["sessions"])
    c5.metric("Mean mastery", counts["mastery"])


def _render_workflow() -> None:
    active = st.session_state.get("active_step", 0)
    cols = st.columns(4)
    for idx, ((name, desc), col) in enumerate(zip(_WORKFLOW, cols, strict=True)):
        with col:
            status = "Complete" if idx < active else "Active" if idx == active else "Pending"
            with st.container(border=True):
                st.caption(f"Step {idx + 1} · {status}")
                st.markdown(f"**{name}**")
                st.write(desc)


def _df(
    frame: pd.DataFrame,
    *,
    height: int = _TABLE_HEIGHT_SMALL,
    column_config: dict | None = None,
) -> None:
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=column_config,
    )


def _topics_df() -> pd.DataFrame:
    kg = st.session_state.get("kg")
    if kg is None:
        return pd.DataFrame(columns=["Topic ID", "Name", "Description", "Mastery"])
    predictor = st.session_state.get("predictor")
    rows = []
    for topic in kg.topics():
        mastery = predictor.mastery(topic.id) if predictor is not None else None
        rows.append(
            {
                "Topic ID": topic.id,
                "Name": topic.name,
                "Description": topic.description,
                "Mastery": f"{mastery:.0%}" if mastery is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _edges_df() -> pd.DataFrame:
    kg = st.session_state.get("kg")
    if kg is None:
        return pd.DataFrame(columns=["Prerequisite", "Unlocks"])
    return pd.DataFrame(
        [{"Prerequisite": edge.source, "Unlocks": edge.target} for edge in kg.edges()]
    )


def _schedule_df() -> pd.DataFrame:
    rows = []
    for session in st.session_state.get("schedule", []):
        rows.append(
            {
                "Date": session.scheduled_date.isoformat(),
                "Topic": session.topic_id or "rest",
                "Action": session.action.value,
                "Minutes": session.duration_minutes,
            }
        )
    return pd.DataFrame(rows)


def _evidence_df() -> pd.DataFrame:
    session = st.session_state.get("session_result")
    if session is None:
        return pd.DataFrame(columns=["Signal", "Value"])
    rows: list[dict[str, str]] = []
    for topic_id, (before, after) in session.mastery_changes.items():
        rows.append(
            {
                "Signal": f"{topic_id} mastery",
                "Value": f"{before:.0%} -> {after:.0%}",
            }
        )
    rows.append({"Signal": "Next action", "Value": session.next_action.value})
    return pd.DataFrame(rows)


def _mastery_df() -> pd.DataFrame:
    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    if kg is None or predictor is None:
        return pd.DataFrame(columns=["Topic", "Mastery", "Status"])
    threshold = load_config()["bkt"]["mastery_threshold"]
    at_risk = load_config()["bkt"]["at_risk_threshold"]
    rows = []
    for topic in kg.topics():
        value = predictor.mastery(topic.id)
        if value >= threshold:
            status = "Mastered"
        elif value < at_risk:
            status = "Needs review"
        else:
            status = "In progress"
        rows.append({"Topic": topic.name, "Mastery": f"{value:.0%}", "Status": status})
    return pd.DataFrame(rows)


def _next_focus() -> tuple[str, str]:
    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    if kg is None or predictor is None:
        return "Create study plan", "Run setup to extract topics and initialise mastery tracking."
    bkt = predictor.all_mastery()
    weakest = min(bkt, key=bkt.get)
    topic = kg.get_topic(weakest)
    return topic.name, f"Current mastery is {bkt[weakest]:.0%}; prioritise this topic next."


def _next_scheduled_session() -> Session | None:
    schedule = sorted(st.session_state.get("schedule", []), key=lambda s: s.scheduled_date)
    return schedule[0] if schedule else None


def _topic_material_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    store = st.session_state.get("store")
    chunks = store.all() if hasattr(store, "all") else []
    for chunk in chunks:
        topic_id = getattr(chunk, "topic_id", None)
        if topic_id:
            counts[topic_id] = counts.get(topic_id, 0) + 1
    if counts:
        return counts
    kg = st.session_state.get("kg")
    if kg is None:
        return {}
    return {topic.id: 0 for topic in kg.topics()}


def _planned_topic_id() -> str | None:
    planned = _next_scheduled_session()
    return planned.topic_id if planned is not None else None


def _next_learning_topic_id() -> str | None:
    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    if kg is None or predictor is None:
        return None
    threshold = load_config()["bkt"]["mastery_threshold"]
    for topic_id in kg.topological_order():
        if predictor.mastery(topic_id) < threshold:
            return topic_id
    return None


def _recommended_topic_id() -> str | None:
    return _planned_topic_id() or _next_learning_topic_id()


def _recommended_action() -> tuple[str, str, str]:
    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    if kg is None or predictor is None:
        return (
            "Start setup",
            "Extract the course knowledge graph",
            "Paste the syllabus and initialise BKT priors before any session work.",
        )

    target_id = _recommended_topic_id()
    target_name = kg.get_topic(target_id).name if target_id else "course"
    material_counts = _topic_material_counts()
    if sum(material_counts.values()) == 0:
        return (
            "Ingest materials",
            f"Attach weekly material for {target_name}",
            "RAG summaries and quizzes need grounded chunks before the session can be credible.",
        )
    if target_id and material_counts.get(target_id, 0) == 0:
        return (
            "Prepare next topic",
            f"Add material for {target_name}",
            "The planner has unlocked this topic, but the evidence base is not loaded yet.",
        )
    if "session_result" not in st.session_state:
        return (
            "Run study session",
            f"Assess {target_name}",
            "Generate quiz evidence, update BKT mastery, and let the scheduler choose the next action.",
        )
    if st.session_state.get("schedule"):
        return (
            "Continue plan",
            f"Next scheduled topic: {target_name}",
            "Run the next session or apply a disruption if the learner is unavailable.",
        )
    return (
        "Course complete",
        "All visible topics are mastered",
        "The current course graph has no remaining low-mastery topic.",
    )


def _quality_gates() -> list[tuple[str, str, str]]:
    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    store = st.session_state.get("store")
    schedule = st.session_state.get("schedule", [])
    session = st.session_state.get("session_result")
    graph_ok = kg is not None and len(kg.topics()) > 0
    rag_ok = store is not None and len(store) > 0
    mastery_ok = predictor is not None and bool(predictor.all_mastery())
    schedule_ok = bool(schedule) or (predictor is not None and all(
        v >= load_config()["bkt"]["mastery_threshold"]
        for v in predictor.all_mastery().values()
    ))
    return [
        ("Ready" if graph_ok else "Waiting", "Knowledge graph", "DAG extracted and prerequisites verified" if graph_ok else "Run setup first"),
        ("Ready" if rag_ok else "Waiting", "RAG evidence", "Material chunks embedded" if rag_ok else "Ingest weekly materials"),
        ("Live" if mastery_ok else "Waiting", "BKT mastery", "Per-topic learner state is active" if mastery_ok else "Initialise priors"),
        ("Synced" if schedule_ok else "Waiting", "Schedule", "Plan follows mastery state" if schedule_ok else "Create an initial plan"),
        ("Captured" if session is not None else "Waiting", "Assessment evidence", "Latest quiz evidence is available" if session is not None else "Run a study session"),
    ]


def _render_decision_panel() -> None:
    label, title, copy = _recommended_action()
    st.markdown(
        f"""
        <div class="decision">
          <div class="decision-label">{label}</div>
          <div class="decision-title">{title}</div>
          <p class="decision-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_quality_gates() -> None:
    items = "\n".join(
        f"""
        <div class="quality-item">
          <div class="quality-state">{state}</div>
          <div class="quality-name">{name}</div>
          <div class="quality-detail">{detail}</div>
        </div>
        """
        for state, name, detail in _quality_gates()
    )
    st.markdown(f'<div class="quality-grid">{items}</div>', unsafe_allow_html=True)


def _advance_schedule_after_session(topic_id: str, next_action: ActionType, cfg: dict) -> None:
    """Mark the completed topic as done and add the next eligible session."""
    schedule = list(st.session_state.get("schedule", []))
    anchor_date = date.today()
    for idx, planned in enumerate(schedule):
        if planned.topic_id == topic_id:
            anchor_date = planned.scheduled_date
            schedule.pop(idx)
            break

    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    if kg is None or predictor is None:
        st.session_state.schedule = schedule
        return

    target = select_topic_for_action(
        next_action,
        candidate_topic_ids=kg.topological_order(),
        bkt_estimates=predictor.all_mastery(),
        kg=kg,
        schedule=schedule,
        config=cfg,
    )
    if target is not None and not any(s.topic_id == target for s in schedule):
        schedule.append(
            Session(
                topic_id=target,
                action=next_action,
                scheduled_date=anchor_date,
                duration_minutes=cfg["scheduler"]["session_duration_minutes"],
            )
        )
    st.session_state.schedule = sorted(schedule, key=lambda s: s.scheduled_date)


def _reconcile_schedule_with_mastery(cfg: dict) -> None:
    """Keep displayed schedule aligned with current mastery state.

    Streamlit state can survive code reloads and user reruns. If a topic is
    already mastered, it should not remain as the next INTRODUCE_NEW item.
    """
    kg = st.session_state.get("kg")
    predictor = st.session_state.get("predictor")
    if kg is None or predictor is None:
        return

    threshold = cfg["bkt"]["mastery_threshold"]
    schedule = list(st.session_state.get("schedule", []))
    retained: list[Session] = []
    removed_dates: list[date] = []
    for planned in schedule:
        if (
            planned.topic_id is not None
            and planned.action == ActionType.INTRODUCE_NEW
            and predictor.mastery(planned.topic_id) >= threshold
        ):
            removed_dates.append(planned.scheduled_date)
            continue
        retained.append(planned)

    if retained != schedule:
        target = select_topic_for_action(
            ActionType.INTRODUCE_NEW,
            candidate_topic_ids=kg.topological_order(),
            bkt_estimates=predictor.all_mastery(),
            kg=kg,
            schedule=retained,
            config=cfg,
        )
        if target is not None and not any(s.topic_id == target for s in retained):
            retained.append(
                Session(
                    topic_id=target,
                    action=ActionType.INTRODUCE_NEW,
                    scheduled_date=min(removed_dates) if removed_dates else date.today(),
                    duration_minutes=cfg["scheduler"]["session_duration_minutes"],
                )
            )
        st.session_state.schedule = sorted(retained, key=lambda s: s.scheduled_date)


def _render_planner_dashboard() -> None:
    top_left, top_right = st.columns([1.08, 0.92], gap="large")
    with top_left:
        _render_decision_panel()
    with top_right, st.container(border=True):
        st.markdown("#### Operational readiness")
        _render_quality_gates()

    focus, rationale = _next_focus()
    left, middle, right = st.columns([1.0, 1.0, 1.0], gap="large")
    with left, st.container(border=True):
        st.markdown("#### Today's focus")
        st.markdown(f"### {focus}")
        st.write(rationale)
    with middle, st.container(border=True):
        st.markdown("#### Learning state")
        mastery = _mastery_df()
        if mastery.empty:
            st.info("No mastery state yet.")
        else:
            _df(
                mastery,
                column_config={
                    "Topic": st.column_config.TextColumn("Topic", width="medium"),
                    "Mastery": st.column_config.TextColumn("Mastery", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="medium"),
                },
            )
    with right, st.container(border=True):
        st.markdown("#### Schedule")
        schedule = _schedule_df()
        if schedule.empty:
            st.info("No sessions planned yet.")
        else:
            _df(
                schedule,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Topic": st.column_config.TextColumn("Topic", width="medium"),
                    "Action": st.column_config.TextColumn("Action", width="medium"),
                    "Minutes": st.column_config.NumberColumn("Minutes", width="small"),
                },
            )

    st.markdown("### Planner workspace")
    c1, c2, c3 = st.columns([1.1, 0.9, 0.9], gap="large")
    with c1, st.container(border=True):
        st.markdown("#### Course structure")
        topics = _topics_df()
        if topics.empty:
            st.write("Start by extracting the course knowledge graph from the Setup tab.")
        else:
            _df(
                topics,
                height=_TABLE_HEIGHT_MEDIUM,
                column_config={
                    "Topic ID": st.column_config.TextColumn("Topic ID", width="medium"),
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "Description": st.column_config.TextColumn("Description", width="medium"),
                    "Mastery": st.column_config.TextColumn("Mastery", width="small"),
                },
            )
    with c2, st.container(border=True):
        st.markdown("#### Recommended workflow")
        _, title, copy = _recommended_action()
        st.markdown(f"**{title}**")
        st.write(copy)
        planned = _next_scheduled_session()
        if planned is not None:
            st.caption(
                f"Current slot: {planned.scheduled_date.isoformat()} · "
                f"{planned.topic_id} · {planned.action.value}"
            )
    with c3, st.container(border=True):
        st.markdown("#### Latest evidence")
        evidence = _evidence_df()
        if evidence.empty:
            st.write("No assessment evidence captured yet.")
        else:
            _df(evidence, height=_TABLE_HEIGHT_SMALL)


def _render_sidebar(cfg: dict) -> None:
    with st.sidebar:
        st.markdown("## AI Study Planner")
        st.caption("Demo operations console")
        st.session_state.provider_name = st.selectbox(
            "LLM provider",
            options=["mock", "anthropic", "ollama"],
            index=0,
            help="Use mock for a deterministic demo without API keys.",
        )
        st.divider()
        st.markdown("### Runtime")
        st.write(f"Seed: `{cfg['random_seed']}`")
        st.write(f"BKT half-life: `{cfg['bkt']['decay']['half_life_days']} days`")
        st.write(f"Mastery threshold: `{cfg['bkt']['mastery_threshold']:.0%}`")
        st.divider()
        if st.button("Reset workspace", use_container_width=True):
            for key in (
                "kg",
                "predictor",
                "store",
                "schedule",
                "weekly_summary",
                "session_result",
                "disruption_result",
                "active_step",
            ):
                st.session_state.pop(key, None)
            st.rerun()


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["random_seed"])
    st.set_page_config(page_title="AI Study Planner", layout="wide")
    _inject_css()
    _render_sidebar(cfg)

    if "store" not in st.session_state:
        st.session_state.store = InMemoryVectorStore()
    if "schedule" not in st.session_state:
        st.session_state.schedule = []
    if "active_step" not in st.session_state:
        st.session_state.active_step = 0
    _reconcile_schedule_with_mastery(cfg)

    _render_header(cfg)
    _render_status_strip()
    _render_workflow()

    tab_planner, tab_setup, tab_weekly, tab_session, tab_disruption = st.tabs(
        ["Planner", "Setup", "Weekly", "Session", "Disruption"]
    )

    with tab_planner:
        _render_planner_dashboard()

    # ----- Setup tab -----
    with tab_setup:
        left, right = st.columns([1.08, 0.92], gap="large")
        with left:
            _panel(
                "Syllabus Intake",
                "Paste a course syllabus or use the included academic demo to initialise the planner.",
            )
            syllabus_text = st.text_area(
                "Syllabus text",
                value=_read_demo_file("syllabus.txt"),
                height=260,
            )
            if st.button(
                "Extract knowledge graph",
                disabled=not syllabus_text,
                type="primary",
                use_container_width=True,
            ):
                result = run_setup(
                    syllabus_text=syllabus_text,
                    provider=_provider(),
                    config=cfg,
                    today=date.today(),
                )
                st.session_state.kg = result.kg
                st.session_state.predictor = result.predictor
                st.session_state.schedule = result.initial_plan
                st.session_state.active_step = max(st.session_state.active_step, 1)
                st.success(
                    f"Extracted {len(result.kg.topics())} topics, "
                    f"{len(result.kg.edges())} prerequisite edges, and "
                    f"{len(result.initial_plan)} initial sessions."
                )
        with right:
            _panel("Knowledge Graph", "Verified topic map and prerequisite chain.")
            _df(
                _topics_df(),
                height=_TABLE_HEIGHT_SMALL,
                column_config={
                    "Topic ID": st.column_config.TextColumn("Topic ID", width="medium"),
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "Description": st.column_config.TextColumn("Description", width="medium"),
                    "Mastery": st.column_config.TextColumn("Mastery", width="small"),
                },
            )
            if not _edges_df().empty:
                _df(_edges_df(), height=112)
            if not _schedule_df().empty:
                st.markdown("**Initial Plan**")
                _df(_schedule_df(), height=112)

    # ----- Weekly tab -----
    with tab_weekly:
        if "kg" not in st.session_state:
            st.info("Complete setup first to create a topic graph.")
        else:
            left, right = st.columns([1.0, 1.0], gap="large")
            with left:
                _panel(
                    "Weekly Materials",
                    "Attach lecture material to the selected topic and generate a grounded summary.",
                )
                week_no = st.number_input("Week number", min_value=1, value=1, step=1)
                topic_options = [t.id for t in st.session_state.kg.topics()]
                default_topic = _recommended_topic_id()
                default_index = topic_options.index(default_topic) if default_topic in topic_options else 0
                sel = st.selectbox(
                    "Topic for these materials",
                    topic_options,
                    index=default_index,
                )
                body = st.text_area(
                    "Material text",
                    value=_demo_material_for(sel),
                    height=230,
                )
                if st.button("Ingest materials", type="primary", use_container_width=True):
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
                    st.session_state.weekly_summary = weekly.summary
                    st.session_state.active_step = max(st.session_state.active_step, 2)
                    st.success(f"Ingested {weekly.chunks_added} material chunks.")
            with right:
                _panel("Weekly Intelligence", "Summary generated from graph-traversal RAG.")
                st.text_area(
                    "Weekly summary",
                    value=st.session_state.get("weekly_summary", ""),
                    height=260,
                    placeholder="Ingest weekly materials to generate a summary.",
                )

    # ----- Session tab -----
    with tab_session:
        if "kg" not in st.session_state or len(st.session_state.store) == 0:
            st.info("Run setup and ingest weekly materials before starting a session.")
        else:
            left, right = st.columns([0.82, 1.18], gap="large")
            with left:
                _panel(
                    "Study Session",
                    "Run a deterministic demo session using reference answers.",
                )
                topic_options = [t.id for t in st.session_state.kg.topics()]
                planned_topic = _recommended_topic_id()
                default_index = topic_options.index(planned_topic) if planned_topic in topic_options else 0
                sel = st.selectbox("Topic", topic_options, index=default_index, key="session_topic")
                material_counts = _topic_material_counts()
                has_material = material_counts.get(sel, 0) > 0
                if not has_material:
                    st.warning(
                        f"No grounded material has been ingested for `{sel}` yet. "
                        "Add weekly material before running this session."
                    )
                if st.button(
                    "Run study session",
                    type="primary",
                    use_container_width=True,
                    disabled=not has_material,
                ):
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
                    st.session_state.session_result = session
                    _advance_schedule_after_session(sel, session.next_action, cfg)
                    st.session_state.active_step = max(st.session_state.active_step, 3)
                    st.success(f"Session complete. Suggested next action: {session.next_action.value}")
                _df(
                    _topics_df(),
                    height=_TABLE_HEIGHT_SMALL,
                    column_config={
                        "Topic ID": st.column_config.TextColumn("Topic ID", width="medium"),
                        "Name": st.column_config.TextColumn("Name", width="medium"),
                        "Description": st.column_config.TextColumn("Description", width="medium"),
                        "Mastery": st.column_config.TextColumn("Mastery", width="small"),
                    },
                )
            with right:
                _panel("Session Evidence", "Quiz grading, mastery movement, and cited explanation.")
                session = st.session_state.get("session_result")
                if session is None:
                    st.caption("Run a study session to populate this panel.")
                else:
                    rows = [
                        {
                            "Question type": g.response.question.type.value,
                            "Score": f"{g.score:.2f}",
                            "Correct": "Yes" if g.correct else "No",
                            "Feedback": g.feedback,
                        }
                        for g in session.graded
                    ]
                    _df(pd.DataFrame(rows), height=142)
                    st.text_area("Cited explanation", value=session.explanation, height=190)

    # ----- Disruption tab -----
    with tab_disruption:
        if "kg" not in st.session_state:
            st.info("Complete setup before applying disruptions.")
        else:
            left, right = st.columns([0.9, 1.1], gap="large")
            with left:
                _panel(
                    "Disruption Report",
                    "Parse natural language into a typed schedule update.",
                )
                text = st.text_area(
                    "What changed?",
                    value="I was sick today and missed my planned regression study session.",
                    placeholder="e.g. I was sick today / deadline moved to ...",
                    height=160,
                )
                if st.button(
                    "Apply disruption",
                    disabled=not text,
                    type="primary",
                    use_container_width=True,
                ):
                    disruption = handle_disruption(
                        report_text=text,
                        schedule=st.session_state.schedule,
                        provider=_provider(),
                        config=cfg,
                    )
                    st.session_state.schedule = disruption.new_schedule
                    st.session_state.disruption_result = disruption
                    st.session_state.active_step = max(st.session_state.active_step, 4)
                    st.success(disruption.confirmation)
            with right:
                _panel("Updated Plan", "Typed update and reflowed schedule.")
                disruption = st.session_state.get("disruption_result")
                if disruption is not None:
                    st.json(
                        {
                            "type": disruption.update.type.value,
                            "payload": disruption.update.payload,
                            "confidence": disruption.update.confidence,
                        }
                    )
                _df(_schedule_df(), height=_TABLE_HEIGHT_SMALL)


if __name__ == "__main__":
    main()
