"""tabint — a deterministic, reproducible intelligence layer for single-table data.

Public API is intentionally small. See docs/architecture.md for the design and
docs/roadmap.md for what's implemented vs. planned.
"""
from tabint.analysis.session import Session
from tabint.shared.results import Result
from tabint.analysis.service.workspace import Workspace, Table
from tabint.analysis.service.relationships import RelationshipGraph, Relationship
from tabint.analysis.db import persistence

__all__ = [
    "Session", "Result", "Workspace", "Table",
    "RelationshipGraph", "Relationship", "persistence",
]
__version__ = "0.1.1"
