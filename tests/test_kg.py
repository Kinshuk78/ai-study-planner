from __future__ import annotations

import pytest

from src.kg import KnowledgeGraph, Topic, ancestors, walk_prerequisites
from src.kg.store import CycleError


def make_kg() -> KnowledgeGraph:
    """linear_algebra -> regression -> logistic_regression
    \\-> neural_networks"""
    kg = KnowledgeGraph()
    for tid in ("linear_algebra", "regression", "logistic_regression", "neural_networks"):
        kg.add_topic(Topic(id=tid, name=tid.replace("_", " ").title()))
    kg.add_prerequisite("linear_algebra", "regression")
    kg.add_prerequisite("regression", "logistic_regression")
    kg.add_prerequisite("regression", "neural_networks")
    return kg


def test_add_topic_and_lookup():
    kg = make_kg()
    assert kg.has_topic("regression")
    assert kg.get_topic("regression").name == "Regression"


def test_duplicate_topic_raises():
    kg = KnowledgeGraph()
    kg.add_topic(Topic(id="x", name="x"))
    with pytest.raises(ValueError):
        kg.add_topic(Topic(id="x", name="x"))


def test_validate_dag_passes_on_valid_graph():
    kg = make_kg()
    kg.validate_dag()  # should not raise


def test_self_loop_raises():
    kg = KnowledgeGraph()
    kg.add_topic(Topic(id="x", name="x"))
    with pytest.raises(CycleError):
        kg.add_prerequisite("x", "x")


def test_cycle_detection_blocks_edge():
    kg = KnowledgeGraph()
    for tid in ("a", "b", "c"):
        kg.add_topic(Topic(id=tid, name=tid))
    kg.add_prerequisite("a", "b")
    kg.add_prerequisite("b", "c")
    with pytest.raises(CycleError):
        kg.add_prerequisite("c", "a")
    # Edge must NOT have been added.
    assert "a" not in kg.get_dependents("c")


def test_unknown_topic_raises():
    kg = KnowledgeGraph()
    kg.add_topic(Topic(id="a", name="a"))
    with pytest.raises(KeyError):
        kg.add_prerequisite("a", "missing")


def test_get_prerequisites_returns_direct_only():
    kg = make_kg()
    assert set(kg.get_prerequisites("logistic_regression")) == {"regression"}
    assert kg.get_prerequisites("linear_algebra") == []


def test_topological_order_respects_edges():
    kg = make_kg()
    order = kg.topological_order()
    assert order.index("linear_algebra") < order.index("regression")
    assert order.index("regression") < order.index("logistic_regression")


def test_traversal_ancestors_excludes_self():
    kg = make_kg()
    anc = ancestors(kg, "logistic_regression")
    assert "logistic_regression" not in anc
    assert set(anc) == {"regression", "linear_algebra"}


def test_traversal_walk_includes_self():
    kg = make_kg()
    walk = walk_prerequisites(kg, "logistic_regression")
    assert walk[0] == "logistic_regression"
    assert set(walk[1:]) == {"regression", "linear_algebra"}


def test_traversal_max_depth():
    kg = make_kg()
    anc = ancestors(kg, "logistic_regression", max_depth=1)
    assert set(anc) == {"regression"}


def test_save_load_roundtrip(tmp_path):
    kg = make_kg()
    path = tmp_path / "kg.json"
    kg.save(path)
    loaded = KnowledgeGraph.load(path)
    assert {t.id for t in loaded.topics()} == {t.id for t in kg.topics()}
    assert {(e.source, e.target) for e in loaded.edges()} == {
        (e.source, e.target) for e in kg.edges()
    }
