"""Extract a knowledge graph from a syllabus."""

from __future__ import annotations

from src.kg import KnowledgeGraph, Topic
from src.llm.json_utils import parse_json_response
from src.llm.provider import LLMProvider


def extract_kg(syllabus_text: str, provider: LLMProvider) -> KnowledgeGraph:
    raw = provider.render_and_complete("KG_EXTRACTION", {"syllabus_text": syllabus_text})
    data = parse_json_response(raw)
    kg = KnowledgeGraph()
    for t in data.get("topics", []):
        kg.add_topic(Topic(id=t["id"], name=t["name"], description=t.get("description", "")))
    for e in data.get("edges", []):
        kg.add_prerequisite(e["source"], e["target"])
    kg.validate_dag()
    return kg
