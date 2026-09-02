"""Foreign-key detection — infer how a workspace's tables connect.

The heuristic is the classic *inclusion dependency* test, run entirely in SQL so
no key columns are pulled into Python:

  A column ``child.fk`` references ``parent.pk`` when
    1. ``parent.pk`` is a primary-key candidate — unique and non-null, and
    2. (almost) every non-null value of ``child.fk`` also appears in
       ``parent.pk`` (coverage ≥ threshold), with a compatible type.

A name match (``fk`` equals ``pk`` or looks like ``<parent>_id``) is recorded as
extra evidence but is not required — the value-level inclusion is what counts.

Works for a single table too: it simply finds no relationships.
"""
from __future__ import annotations

from typing import Any

import ibis.expr.datatypes as dt
from pydantic import BaseModel, Field

# A child column qualifies as a foreign key when at least this fraction of its
# distinct non-null values are found in the parent key.
_COVERAGE_THRESHOLD = 0.95
# Require at least this many distinct non-null values, so a column with one or
# two repeated values doesn't spuriously "include" into some key.
_MIN_DISTINCT = 2


class Relationship(BaseModel):
    """A detected foreign-key edge: child.column → parent.column."""
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    coverage: float          # fraction of child values found in the parent key
    name_match: bool         # column names also line up (extra evidence)

    def __repr__(self) -> str:
        star = " ~name" if self.name_match else ""
        return (
            f"{self.child_table}.{self.child_column} → "
            f"{self.parent_table}.{self.parent_column} "
            f"({self.coverage:.0%}{star})"
        )


class RelationshipGraph(BaseModel):
    """The connection graph over a workspace's tables.

    Nodes are tables (with row counts); edges are detected foreign keys.
    """
    tables: dict[str, int] = Field(default_factory=dict)          # name → row count
    relationships: list[Relationship] = Field(default_factory=list)

    def neighbors(self, table: str) -> set[str]:
        """Tables directly connected to ``table`` in either direction."""
        out = set()
        for r in self.relationships:
            if r.child_table == table:
                out.add(r.parent_table)
            elif r.parent_table == table:
                out.add(r.child_table)
        return out

    def __repr__(self) -> str:
        if not self.relationships:
            return f"<RelationshipGraph tables={list(self.tables)} (no links found)>"
        lines = "\n".join(f"  {r!r}" for r in self.relationships)
        return f"<RelationshipGraph {len(self.tables)} tables, {len(self.relationships)} links>\n{lines}"


def detect_relationships(workspace: Any) -> RelationshipGraph:
    """Scan every table pair for foreign-key inclusion dependencies."""
    tables = workspace.table_names
    graph = RelationshipGraph()

    # Per-table column stats and row counts (one pass each).
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for name in tables:
        table = workspace.table(name)
        schema = table._table.schema()
        n_rows = int(workspace.run_sql(f"SELECT COUNT(*) AS n FROM {name}")["n"].iloc[0])
        graph.tables[name] = n_rows
        stats[name] = _column_stats(workspace, name, schema, n_rows)

    # Primary-key candidates: unique, non-null, key-typed columns.
    pk_candidates = {
        name: {c: s for c, s in cols.items() if s["is_pk_candidate"]}
        for name, cols in stats.items()
    }

    for child in tables:
        for fk_col, fk_stat in stats[child].items():
            if fk_stat["kind"] not in ("int", "str"):
                continue
            if fk_stat["n_distinct"] < _MIN_DISTINCT:
                continue
            # A column that is its own table's primary key is normally NOT a
            # foreign key — two unrelated surrogate id ranges (1..100 ⊂ 1..500)
            # would otherwise pass the inclusion test and fabricate an edge. Only
            # accept such a column if its name specifically points at the parent
            # (e.g. user_id → users), which is the genuine 1:1 case.
            child_is_pk = fk_stat["is_pk_candidate"]
            for parent in tables:
                for pk_col, pk_stat in pk_candidates[parent].items():
                    if child == parent and fk_col == pk_col:
                        continue  # a key can't be a foreign key to itself
                    if pk_stat["kind"] != fk_stat["kind"]:
                        continue
                    if child_is_pk and not _strong_name_match(fk_col, parent):
                        continue
                    coverage = _coverage(workspace, child, fk_col, parent, pk_col)
                    if coverage >= _COVERAGE_THRESHOLD:
                        graph.relationships.append(Relationship(
                            child_table=child,
                            child_column=fk_col,
                            parent_table=parent,
                            parent_column=pk_col,
                            coverage=round(coverage, 4),
                            name_match=_name_match(fk_col, pk_col, parent),
                        ))
    return graph


def find_join_edge(
    graph: RelationshipGraph, joined: list[str], candidate: str
) -> tuple[str, str, str] | None:
    """Find an FK edge connecting ``candidate`` to any already-joined table.

    Returns (left_table, left_column, right_column) suitable for an ON clause,
    or None if no relationship links them. Works in either direction.
    """
    joined_set = set(joined)
    for r in graph.relationships:
        if r.child_table in joined_set and r.parent_table == candidate:
            return (r.child_table, r.child_column, r.parent_column)
        if r.parent_table in joined_set and r.child_table == candidate:
            return (r.parent_table, r.parent_column, r.child_column)
    return None


def _column_stats(
    workspace: Any, table: str, schema: Any, n_rows: int
) -> dict[str, dict[str, Any]]:
    """Row count, non-null count, distinct count, kind, and PK-candidacy per column."""
    out: dict[str, dict[str, Any]] = {}
    for col in schema.names:
        kind = _kind(schema[col])
        if kind not in ("int", "str"):
            out[col] = {"kind": kind, "n_distinct": 0, "is_pk_candidate": False}
            continue
        row = workspace.run_sql(
            f'SELECT COUNT("{col}") AS nn, COUNT(DISTINCT "{col}") AS d FROM {table}'
        )
        nn, d = int(row["nn"].iloc[0]), int(row["d"].iloc[0])
        out[col] = {
            "kind": kind,
            "n_distinct": d,
            # Unique and non-null over all rows → a valid key to be referenced.
            "is_pk_candidate": n_rows > 0 and nn == n_rows and d == n_rows,
        }
    return out


def _coverage(workspace: Any, child: str, fk: str, parent: str, pk: str) -> float:
    """Fraction of child.fk distinct non-null values present in parent.pk."""
    row = workspace.run_sql(
        f'''
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE p.v IS NULL) AS missing
        FROM (SELECT DISTINCT "{fk}" AS v FROM {child} WHERE "{fk}" IS NOT NULL) c
        LEFT JOIN (SELECT DISTINCT "{pk}" AS v FROM {parent}) p ON c.v = p.v
        '''
    )
    total, missing = int(row["total"].iloc[0]), int(row["missing"].iloc[0])
    return (total - missing) / total if total > 0 else 0.0


def _name_match(fk: str, pk: str, parent: str) -> bool:
    """Broad name evidence: identical column name, or looks like ``<parent>_id``."""
    fk_l, pk_l = fk.lower(), pk.lower()
    return fk_l == pk_l or _strong_name_match(fk, parent)


def _strong_name_match(fk: str, parent: str) -> bool:
    """Parent-specific name evidence: ``<parent>_id`` / ``<parent-singular>_id``.

    Deliberately excludes the generic ``id == id`` case, so an unrelated surrogate
    key named ``id`` does not count as pointing at another table's ``id``.
    """
    fk_l, parent_l = fk.lower(), parent.lower()
    singular = parent_l[:-1] if parent_l.endswith("s") else parent_l
    return fk_l in {f"{parent_l}_id", f"{singular}_id"}


def _kind(dtype: Any) -> str:
    if isinstance(dtype, dt.Integer):
        return "int"
    if isinstance(dtype, dt.String):
        return "str"
    if isinstance(dtype, (dt.Floating, dt.Decimal)):
        return "float"
    if isinstance(dtype, dt.Boolean):
        return "bool"
    return "other"
