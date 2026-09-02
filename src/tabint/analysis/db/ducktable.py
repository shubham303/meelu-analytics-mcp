"""Low-level DuckDB table mechanics shared by Store and Workspace/Table.

Both the single-table Store and the multi-table Workspace need the same three
primitives: load a CSV into a table stamped with a stable row id, read it back in
that order, and write a computed column back by position. Keeping that logic here
means there is exactly one implementation of the ``_ti_row`` machinery.

Every table is stored as:
  - an internal table ``<internal>`` carrying a 0-based ``_ti_row`` id column
  - a view ``<view>`` = the internal table minus ``_ti_row`` (what callers query)

All identifiers (table, view, column names) are double-quoted before being spliced
into SQL, so names containing spaces, punctuation, or reserved words are handled
safely; the CSV path is single-quote-escaped for the same reason.

All functions take an ibis DuckDB backend; ``backend.con`` is the underlying
duckdb connection used for the low-level writes ibis doesn't cover.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def quote_ident(name: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    """Escape a value for use inside a single-quoted SQL string literal."""
    return value.replace("'", "''")


def load_csv(backend: Any, csv_path: Path, internal: str, view: str) -> None:
    """Create the internal table (with _ti_row) and its public view from a CSV."""
    path_literal = _quote_literal(str(csv_path))
    create_from_query(backend, f"SELECT * FROM read_csv_auto('{path_literal}')", internal, view)


def create_from_query(backend: Any, select_sql: str, internal: str, view: str) -> None:
    """Materialize a SELECT as a new internal table (with _ti_row) plus its view.

    Used for CSV loads and for derived tables such as joins — anything that
    produces rows and needs the same stable-row-id + hidden-view treatment.
    """
    qi, qv = quote_ident(internal), quote_ident(view)
    con = backend.con
    con.execute(
        f"CREATE TABLE {qi} AS "
        f"SELECT row_number() OVER () - 1 AS _ti_row, * FROM ({select_sql})"
    )
    con.execute(f"CREATE VIEW {qv} AS SELECT * EXCLUDE (_ti_row) FROM {qi}")


def create_empty(backend: Any, columns: list[tuple[str, str]], internal: str, view: str) -> None:
    """Create an empty structured table (with auto-assigned _ti_row) plus its view.

    Unlike create_from_query, this table starts with no rows: it defines a schema
    the caller then fills with insert_select. A DuckDB sequence supplies _ti_row
    so ordinary INSERTs that omit it get a monotonically increasing id for free,
    keeping the table compatible with the frame_in_order / write_back machinery.

    Args:
        columns: (name, sql_type) pairs. Types must already be validated.
    """
    qi, qv = quote_ident(internal), quote_ident(view)
    seq = quote_ident(f"_seq_{internal}")
    seq_literal = _quote_literal(f"_seq_{internal}")
    con = backend.con
    coldefs = ", ".join(f"{quote_ident(name)} {sql_type}" for name, sql_type in columns)
    con.execute(f"CREATE SEQUENCE {seq}")
    con.execute(
        f"CREATE TABLE {qi} (_ti_row BIGINT DEFAULT nextval('{seq_literal}'), {coldefs})"
    )
    con.execute(f"CREATE VIEW {qv} AS SELECT * EXCLUDE (_ti_row) FROM {qi}")


def insert_select(backend: Any, internal: str, source_sql: str) -> int:
    """Append rows into an existing table from a SELECT/VALUES query; return the count.

    The column list (every column except the auto-assigned _ti_row) is spliced in
    explicitly so source_sql maps positionally to the table's own columns and the
    sequence default fills _ti_row. Returns the number of rows inserted.
    """
    qi = quote_ident(internal)
    con = backend.con
    cols = [row[0] for row in con.execute(f"DESCRIBE {qi}").fetchall() if row[0] != "_ti_row"]
    col_list = ", ".join(quote_ident(c) for c in cols)
    before = con.execute(f"SELECT COUNT(*) FROM {qi}").fetchone()[0]
    con.execute(f"INSERT INTO {qi} ({col_list}) {source_sql}")
    after = con.execute(f"SELECT COUNT(*) FROM {qi}").fetchone()[0]
    return int(after - before)


# Persistent registry of derived (computed) columns — cluster labels, outlier
# flags, predictions, etc. — that must be excluded from feature matrices. It
# lives in the same DuckDB file as the data so it survives reopen/_reattach; keyed
# by (table_key, column_name) where table_key is the public table/view name.
_DERIVED_REGISTRY = "_ti_derived_columns"


def _ensure_derived_registry(con: Any) -> None:
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {quote_ident(_DERIVED_REGISTRY)} "
        f"(table_key VARCHAR, column_name VARCHAR)"
    )


def register_derived(backend: Any, table_key: str, column_name: str) -> None:
    """Mark a column as derived (a non-feature annotation) for the given table."""
    con = backend.con
    q = quote_ident(_DERIVED_REGISTRY)
    _ensure_derived_registry(con)
    con.execute(
        f"DELETE FROM {q} WHERE table_key = ? AND column_name = ?", [table_key, column_name]
    )
    con.execute(f"INSERT INTO {q} VALUES (?, ?)", [table_key, column_name])


def unregister_derived(backend: Any, table_key: str, column_name: str) -> None:
    """Clear any derived mark on a column (e.g. when it is (re)written as a feature)."""
    con = backend.con
    _ensure_derived_registry(con)
    con.execute(
        f"DELETE FROM {quote_ident(_DERIVED_REGISTRY)} "
        f"WHERE table_key = ? AND column_name = ?",
        [table_key, column_name],
    )


def derived_columns(backend: Any, table_key: str) -> set[str]:
    """Return the set of columns marked derived for the given table."""
    con = backend.con
    _ensure_derived_registry(con)
    rows = con.execute(
        f"SELECT column_name FROM {quote_ident(_DERIVED_REGISTRY)} WHERE table_key = ?",
        [table_key],
    ).fetchall()
    return {r[0] for r in rows}


# Persistent store of per-column types. Mirrors _DERIVED_REGISTRY: a sidecar
# table in the same DuckDB file, keyed by (table_key, column_name), so it
# survives reopen/_reattach for free (no persistence.py code). Unlike the
# derived registry this one holds a value (the type string), and it is the
# single source of truth for a column's type: classify_column reads it first
# and writes back into it (cache-and-compute), and set_column_type (the LLM
# tool) writes a refined value into it.
_TYPE_REGISTRY = "_ti_column_types"


def _ensure_type_registry(con: Any) -> None:
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {quote_ident(_TYPE_REGISTRY)} "
        f"(table_key VARCHAR, column_name VARCHAR, type VARCHAR)"
    )


def get_column_type(
    backend: Any, table_key: str, column_name: str
) -> str | None:
    """Return the cached type for one column, or None if it has none yet."""
    con = backend.con
    _ensure_type_registry(con)
    row = con.execute(
        f"SELECT type FROM {quote_ident(_TYPE_REGISTRY)} "
        f"WHERE table_key = ? AND column_name = ?",
        [table_key, column_name],
    ).fetchone()
    return row[0] if row else None


def set_column_type(
    backend: Any, table_key: str, column_name: str, type_value: str
) -> None:
    """Upsert a column's type (DELETE then INSERT — idempotent on re-set)."""
    con = backend.con
    _ensure_type_registry(con)
    con.execute(
        f"DELETE FROM {quote_ident(_TYPE_REGISTRY)} "
        f"WHERE table_key = ? AND column_name = ?",
        [table_key, column_name],
    )
    con.execute(
        f"INSERT INTO {quote_ident(_TYPE_REGISTRY)} VALUES (?, ?, ?)",
        [table_key, column_name, type_value],
    )


def unset_column_type(
    backend: Any, table_key: str, column_name: str
) -> None:
    """Clear a column's cached type so the next classify_column recomputes it."""
    con = backend.con
    _ensure_type_registry(con)
    con.execute(
        f"DELETE FROM {quote_ident(_TYPE_REGISTRY)} "
        f"WHERE table_key = ? AND column_name = ?",
        [table_key, column_name],
    )


def column_types(backend: Any, table_key: str) -> dict[str, str]:
    """Return {column_name: type} for every typed column of the given table."""
    con = backend.con
    _ensure_type_registry(con)
    rows = con.execute(
        f"SELECT column_name, type FROM {quote_ident(_TYPE_REGISTRY)} WHERE table_key = ?",
        [table_key],
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# Tokens that must never appear in a feature expression: statement/DDL keywords,
# subquery starters, and functions that reach the filesystem or catalog. Matched
# as whole words, case-insensitively — the guard for agent-authored SQL that runs
# server-side over massive tables (see add_computed_column).
_SQL_EXPR_BLOCKLIST = frozenset({
    "select", "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "copy", "install", "load", "pragma", "call", "export", "import",
    "set", "reset", "read_csv", "read_csv_auto", "read_parquet", "read_json",
    "read_json_auto", "read_text", "read_blob", "glob", "system", "getvariable",
})
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _validate_sql_expression(backend: Any, view: str, expression: str) -> None:
    """Reject anything beyond a single scalar expression over the table's columns.

    Layered guard: no statement chaining (``;``), no blocklisted keywords/functions
    (subqueries, DDL/DML, file & catalog readers), then a zero-row bind check so
    the expression must parse as a scalar over the real columns. Raises ValueError.
    """
    if not expression or not expression.strip():
        raise ValueError("Feature expression is empty.")
    if ";" in expression:
        raise ValueError("Feature expression must be a single expression (no ';').")
    lowered_words = {w.lower() for w in _WORD_RE.findall(expression)}
    hit = lowered_words & _SQL_EXPR_BLOCKLIST
    if hit:
        raise ValueError(
            f"Feature expression may not use {sorted(hit)} — only a scalar computation "
            f"over the table's columns is allowed (no subqueries, DDL, or file access)."
        )
    # Bind-only check: WHERE false computes nothing but still parses/binds the
    # expression and its column references. Cheap even on a huge table.
    qv = quote_ident(view)
    try:
        backend.con.execute(f"SELECT ({expression}) FROM {qv} WHERE false")
    except Exception as exc:
        raise ValueError(f"Invalid feature expression: {exc}") from exc


def add_computed_column(backend: Any, internal: str, view: str, name: str, expression: str) -> int:
    """Add (or replace) a column computed by a SQL scalar expression, in-database.

    Rebuilds the internal table with the expression as a new column and refreshes
    the public view, preserving ``_ti_row``. Runs entirely inside DuckDB — no rows
    are materialized in the app. Returns the count of non-null values produced.
    """
    _validate_sql_expression(backend, view, expression)
    con = backend.con
    qi, qv, qn = quote_ident(internal), quote_ident(view), quote_ident(name)
    existing = {row[0] for row in con.execute(f"DESCRIBE {qi}").fetchall()}
    keep = f"* EXCLUDE ({qn})" if name in existing else "*"
    con.execute(f"CREATE OR REPLACE TABLE {qi} AS SELECT {keep}, ({expression}) AS {qn} FROM {qi}")
    con.execute(f"CREATE OR REPLACE VIEW {qv} AS SELECT * EXCLUDE (_ti_row) FROM {qi}")
    return int(con.execute(f"SELECT COUNT({qn}) FROM {qi}").fetchone()[0])


def count_rows(backend: Any, view: str) -> int:
    """Return the table's row count via an in-database COUNT(*) (no materialization)."""
    return int(backend.con.execute(f"SELECT COUNT(*) FROM {quote_ident(view)}").fetchone()[0])


def count_non_null(backend: Any, view: str, column: str) -> int:
    """Return the count of non-NULL values in a column, computed inside DuckDB.

    Raises KeyError if the column does not exist (COUNT(col) counts non-NULLs).
    """
    con = backend.con
    qv = quote_ident(view)
    existing = {row[0] for row in con.execute(f"DESCRIBE {qv}").fetchall()}
    if column not in existing:
        raise KeyError(f"No column {column!r}. Known: {sorted(existing)}")
    return int(con.execute(f"SELECT COUNT({quote_ident(column)}) FROM {qv}").fetchone()[0])


def frame_in_order(backend: Any, internal: str) -> Any:
    """Return the table as a pandas DataFrame in stable _ti_row order (id excluded)."""
    return backend.sql(
        f"SELECT * EXCLUDE (_ti_row) FROM {quote_ident(internal)} ORDER BY _ti_row"
    ).execute()


def write_back(backend: Any, internal: str, view: str, name: str, values: Any) -> None:
    """Add or replace a column by position (join on _ti_row), then refresh the view.

    ``values[i]`` is written to the row whose ``_ti_row`` is ``i`` — the same
    order frame_in_order returns, so per-row arrays align without realignment.

    Raises ValueError if len(values) != the table's row count, since a mismatch
    would silently drop rows through the positional inner join.
    """
    # DuckDB parameter binding can't consume numpy generics — coerce to Python.
    col_list = [v.item() if isinstance(v, np.generic) else v for v in values]
    n = len(col_list)
    con = backend.con
    qi, qv, qn = quote_ident(internal), quote_ident(view), quote_ident(name)
    tmp = quote_ident(f"_wb_{internal}")

    current = con.execute(f"SELECT COUNT(*) FROM {qi}").fetchone()[0]
    if n != current:
        raise ValueError(
            f"write_back_column expected {current} values (one per row) but got {n}."
        )

    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {tmp} AS "
        f"SELECT unnest(range({n})) AS _ti_row, unnest(?) AS {qn}",
        [col_list],
    )
    existing = {row[0] for row in con.execute(f"DESCRIBE {qi}").fetchall()}
    src_cols = f"{qi}.* EXCLUDE ({qn})" if name in existing else f"{qi}.*"

    con.execute(
        f"CREATE OR REPLACE TABLE {qi} AS "
        f"SELECT {src_cols}, {tmp}.{qn} "
        f"FROM {qi} JOIN {tmp} ON {qi}._ti_row = {tmp}._ti_row"
    )
    con.execute(f"CREATE OR REPLACE VIEW {qv} AS SELECT * EXCLUDE (_ti_row) FROM {qi}")
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
