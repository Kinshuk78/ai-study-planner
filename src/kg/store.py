"""NetworkX-backed knowledge graph store.

Invariant: the graph is a DAG with a single ``is_prerequisite_of`` edge
type. Call :meth:`KnowledgeGraph.validate_dag` after every construction
or mutation that adds edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.kg.schema import KGEdge, Topic


class CycleError(ValueError):
    """Raised when an edge would introduce a cycle."""


class KnowledgeGraph:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ----- mutation -------------------------------------------------

    def add_topic(self, topic: Topic) -> None:
        if topic.id in self._graph:
            raise ValueError(f"topic '{topic.id}' already exists")
        self._graph.add_node(topic.id, name=topic.name, description=topic.description)

    def add_prerequisite(self, source_id: str, target_id: str) -> None:
        """Adds the edge ``source --is_prerequisite_of--> target``.

        Raises :class:`CycleError` if the edge would introduce a cycle.
        """
        if source_id not in self._graph:
            raise KeyError(f"unknown source topic '{source_id}'")
        if target_id not in self._graph:
            raise KeyError(f"unknown target topic '{target_id}'")
        if source_id == target_id:
            raise CycleError(f"self-loop on '{source_id}'")
        self._graph.add_edge(source_id, target_id, edge_type="is_prerequisite_of")
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(source_id, target_id)
            raise CycleError(f"adding edge {source_id} -> {target_id} would create a cycle")

    # ----- queries --------------------------------------------------

    def has_topic(self, topic_id: str) -> bool:
        return topic_id in self._graph

    def get_topic(self, topic_id: str) -> Topic:
        attrs = self._graph.nodes[topic_id]
        return Topic(id=topic_id, name=attrs["name"], description=attrs.get("description", ""))

    def topics(self) -> list[Topic]:
        return [self.get_topic(tid) for tid in self._graph.nodes]

    def edges(self) -> list[KGEdge]:
        return [
            KGEdge(source=u, target=v, edge_type=data.get("edge_type", "is_prerequisite_of"))
            for u, v, data in self._graph.edges(data=True)
        ]

    def get_prerequisites(self, topic_id: str) -> list[str]:
        """Direct prerequisites (one hop)."""
        return list(self._graph.predecessors(topic_id))

    def get_dependents(self, topic_id: str) -> list[str]:
        """Direct dependents (one hop)."""
        return list(self._graph.successors(topic_id))

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._graph))

    @property
    def graph(self) -> nx.DiGraph:
        """Underlying NetworkX graph (read-only intent)."""
        return self._graph

    # ----- validation -----------------------------------------------

    def validate_dag(self) -> None:
        """Raises :class:`CycleError` if the graph is not a DAG."""
        if not nx.is_directed_acyclic_graph(self._graph):
            cycle = nx.find_cycle(self._graph)
            raise CycleError(f"graph contains cycle: {cycle}")

    # ----- persistence ----------------------------------------------

    def to_dict(self) -> dict:
        return {
            "topics": [
                {"id": t.id, "name": t.name, "description": t.description} for t in self.topics()
            ],
            "edges": [{"source": e.source, "target": e.target} for e in self.edges()],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        kg = cls()
        for t in data.get("topics", []):
            kg.add_topic(Topic(id=t["id"], name=t["name"], description=t.get("description", "")))
        for e in data.get("edges", []):
            kg.add_prerequisite(e["source"], e["target"])
        kg.validate_dag()
        return kg

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        return cls.from_dict(json.loads(Path(path).read_text()))
