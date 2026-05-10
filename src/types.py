"""Shared dataclasses used across workstreams.

These are the stable interfaces between modules. Changes to anything in
this file require coordination with the owning workstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

# ---------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Topic:
    id: str  # snake_case
    name: str
    description: str = ""


@dataclass(frozen=True)
class KGEdge:
    source: str  # topic id
    target: str  # topic id
    edge_type: str = "is_prerequisite_of"


# ---------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------


class ActionType(str, Enum):
    INTRODUCE_NEW = "INTRODUCE_NEW"
    REVIEW_WEAKEST = "REVIEW_WEAKEST"
    QUIZ_EXISTING = "QUIZ_EXISTING"
    REST = "REST"


@dataclass
class SchedulerState:
    """Discretised state used by Q-learning. Raw values are kept for
    debugging; the encoder collapses them to a single state id."""

    fraction_mastered: float
    days_remaining: int
    num_at_risk: int
    last_action: ActionType


@dataclass
class Session:
    topic_id: str | None  # None for REST
    action: ActionType
    scheduled_date: date
    duration_minutes: int


# ---------------------------------------------------------------------
# Quiz / grading
# ---------------------------------------------------------------------


class QuestionType(str, Enum):
    MCQ = "mcq"
    SHORT = "short"


@dataclass
class QuizQuestion:
    stem: str
    answer: str
    type: QuestionType
    choices: list[str] = field(default_factory=list)  # only for MCQ
    citations: list[str] = field(default_factory=list)  # chunk ids


@dataclass
class QuizResponse:
    question: QuizQuestion
    student_answer: str


@dataclass
class GradedResponse:
    response: QuizResponse
    score: float  # 0.0, 0.5, or 1.0
    correct: bool
    feedback: str = ""


# ---------------------------------------------------------------------
# Disruption updates
# ---------------------------------------------------------------------


class DisruptionType(str, Enum):
    SICK_DAY = "sick_day"
    DEADLINE_CHANGE = "deadline_change"
    CAPACITY_CHANGE = "capacity_change"
    COMPLETED_EXTERNALLY = "completed_externally"


@dataclass
class DisruptionUpdate:
    type: DisruptionType
    payload: dict
    confidence: float = 1.0


# ---------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------


@dataclass
class Chunk:
    id: str  # uuid
    text: str
    source: str  # filename or document id
    page: int | None = None
    topic_id: str | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
