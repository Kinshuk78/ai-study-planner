"""KG schema. Re-exports the canonical dataclasses from :mod:`src.types`.

Workstream 1 owns this file. Other modules should import from
``src.kg`` rather than ``src.types`` so the dependency direction stays
``everything -> kg -> types``.
"""

from src.types import KGEdge, Topic

__all__ = ["KGEdge", "Topic"]
