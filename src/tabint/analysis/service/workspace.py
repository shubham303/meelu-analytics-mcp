"""Workspace — an in-memory DuckDB database holding one or more related tables.

A Workspace owns a single DuckDB connection and registers each uploaded CSV as
its own named table (each stamped with a ``_ti_row`` id, exposed through a view).
Because every table lives in one connection, foreign-key detection and future
cross-table joins are ordinary SQL — no materialising key columns into Python.

Each Workspace is **independent and per-session**: it uses its own in-memory
DuckDB database, so trained models and derived (write-back) columns never leak
between two Sessions — even when they load the same file.

A single-table session is just a Workspace with one table, so single- and
multi-table paths share the same code.

``Table`` is the per-table handle. It carries the analytics API (profile,
cluster, train_classifier, …); each method delegates to the analytics layer,
passing itself as the store-like object those functions expect (it exposes
``get_frame``, ``write_back_column``, ``run_sql`` and the ibis ``_table``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..db import ducktable as _ducktable
from .algorithms import (
    association,
    basket,
    causal,
    clustering,
    cohort,
    compare,
    descriptive,
    dimreduction,
    feature_computation,
    insights,
    interpretation,
    supervised,
    timeseries,
)
from ...shared.identity import _lazy_import
from .relationships import find_join_edge
from ...shared.results import Result

ibis = _lazy_import("ibis")

# SQL reserved words that _sanitize must not emit as a bare identifier.
_RESERVED = frozenset({
    "all", "and", "any", "array", "as", "asc", "both", "case", "cast", "check",
    "collate", "column", "constraint", "create", "cross", "default", "desc",
    "distinct", "do", "else", "end", "except", "false", "fetch", "for", "foreign",
    "from", "full", "grant", "group", "having", "in", "inner", "intersect", "into",
    "is", "join", "lateral", "leading", "left", "like", "limit", "natural", "not",
    "null", "offset", "on", "only", "or", "order", "outer", "pivot", "primary",
    "references", "returning", "right", "select", "some", "table", "then", "to",
    "true", "union", "unique", "unpivot", "using", "user", "values", "view",
    "when", "where", "window", "with",
})


# DuckDB column types accepted by create_table's schema. Each may carry a
# parenthesised precision/length (e.g. DECIMAL(10,2), VARCHAR(50)); the regex
# below enforces that shape so a "type" can never smuggle in arbitrary DDL.
_ALLOWED_TYPES = frozenset({
    "boolean", "bool", "tinyint", "smallint", "integer", "int", "bigint",
    "hugeint", "utinyint", "usmallint", "uinteger", "ubigint", "real", "float",
    "double", "decimal", "numeric", "varchar", "char", "text", "string", "blob",
    "date", "time", "timestamp", "timestamptz", "interval", "uuid", "json",
})
_TYPE_RE = re.compile(r"^([a-z]+)(\s*\(\s*\d+\s*(?:,\s*\d+\s*)?\))?$", re.IGNORECASE)


def _validate_type(sql_type: str) -> str:
    """Validate an agent-supplied column type against the allowlist; return it.

    Guards the one spot where caller text is spliced into DDL. Raises ValueError
    on anything that is not a known base type optionally followed by a numeric
    precision, so create_table can never execute injected SQL.
    """
    match = _TYPE_RE.match(sql_type.strip())
    if not match or match.group(1).lower() not in _ALLOWED_TYPES:
        raise ValueError(
            f"Unsupported column type {sql_type!r}. Allowed: {sorted(_ALLOWED_TYPES)} "
            f"(optionally with a numeric precision, e.g. DECIMAL(10,2))."
        )
    return sql_type.strip()


def _sanitize(name: str) -> str:
    """Turn a filename stem into a safe, non-reserved SQL identifier."""
    cleaned = re.sub(r"\W+", "_", name).strip("_").lower()
    if not cleaned:
        cleaned = "tbl"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    if cleaned in _RESERVED:
        cleaned = f"{cleaned}_tbl"
    return cleaned


class Table:
    """Handle to one table within a Workspace, plus the analytics it can run.

    The low-level methods (get_frame / write_back_column / run_sql / _table) make
    this object quack like the old single-table Store, so every analytics
    function works against it unchanged.
    """

    def __init__(self, workspace: "Workspace", name: str, internal: str, view: str) -> None:
        self._ws = workspace
        self.name = name
        self._internal = internal
        self._view = view
        self._table = workspace._ibis.table(view)  # ibis TableExpr for the view
        self.models: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"<Table {self.name!r} cols={list(self._table.schema().names)}>"

    # --- low-level (store-compatible) ------------------------------------- #

    def get_frame(self) -> Any:
        """Full table as a pandas DataFrame in stable _ti_row order."""
        return _ducktable.frame_in_order(self._ws._ibis, self._internal)

    def count_rows(self) -> int:
        """Row count via an in-database COUNT(*) — cheap, no rows materialized."""
        return _ducktable.count_rows(self._ws._ibis, self._view)

    def count_non_null(self, column: str) -> int:
        """Count of non-NULL values in a column, computed inside DuckDB."""
        return _ducktable.count_non_null(self._ws._ibis, self._view, column)

    def run_sql(self, query: str) -> Any:
        """Run SQL across the whole workspace (all tables are visible by name)."""
        return self._ws.run_sql(query)

    def write_back_column(self, name: str, values: Any, feature: bool = False) -> None:
        """Add or replace a column by row position (see _ducktable.write_back).

        ``feature`` declares intent for downstream modelling. Written-back columns
        are computed annotations (outlier flags, cluster labels, predictions) and
        default to ``feature=False`` — recorded in the derived-column registry and
        skipped by feature_columns so they never leak into a feature matrix. Pass
        ``feature=True`` for derived columns that ARE meant to be features (e.g. the
        components from reduce_dimensions, to cluster/train on afterwards).

        If the column already existed, its cached type is cleared so the next
        classify_column recomputes it — the data has changed, so the old type may
        no longer be right.
        """
        self._invalidate_column_type_if_exists(name)
        _ducktable.write_back(self._ws._ibis, self._internal, self._view, name, values)
        self._table = self._ws._ibis.table(self._view)
        if feature:
            _ducktable.unregister_derived(self._ws._ibis, self.name, name)
        else:
            _ducktable.register_derived(self._ws._ibis, self.name, name)

    def add_computed_column(self, name: str, expression: str, feature: bool = True) -> int:
        """Materialize a SQL scalar expression as a new column, in-database.

        The computation runs inside DuckDB over the stored table (nothing is pulled
        into the app), so it scales to arbitrarily large tables. The expression is
        validated to be a single scalar over the table's columns. Returns the count
        of non-null values. ``feature`` controls registry membership exactly like
        write_back_column (default True = a real, model-eligible feature).

        If the column already existed, its cached type is cleared (see write_back_column).
        """
        self._invalidate_column_type_if_exists(name)
        n = _ducktable.add_computed_column(self._ws._ibis, self._internal, self._view, name, expression)
        self._table = self._ws._ibis.table(self._view)
        if feature:
            _ducktable.unregister_derived(self._ws._ibis, self.name, name)
        else:
            _ducktable.register_derived(self._ws._ibis, self.name, name)
        return n

    def derived_columns(self) -> set[str]:
        """Names of this table's derived (non-feature) columns; see write_back_column."""
        return _ducktable.derived_columns(self._ws._ibis, self.name)

    # --- column-type cache (the sidecar type store) ---------------------- #
    #
    # Two layers over _ducktable's _ti_column_types table:
    #   * the *_cached_* accessors are unvalidated — the cache reads/writes that
    #     dtypes.classify_column (cache-and-compute) uses for ANY type, including
    #     auto-detected continuous/datetime/identifier.
    #   * set_column_type is the validated, agent-facing method: it accepts only
    #     categorical_nominal / categorical_ordinal (the refinements) and rejects
    #     everything else, then writes through to the same cache.

    def get_cached_type(self, column: str) -> str | None:
        """Cached type for one column, or None if it has not been classified yet."""
        return _ducktable.get_column_type(self._ws._ibis, self.name, column)

    def set_cached_type(self, column: str, type_value: str) -> None:
        """Write a type into the cache directly (no validation). Internal use by
        classify_column — accepts any value in COLUMN_TYPES."""
        _ducktable.set_column_type(self._ws._ibis, self.name, column, type_value)

    def unset_cached_type(self, column: str) -> None:
        """Clear a column's cached type so the next classify_column recomputes it."""
        _ducktable.unset_column_type(self._ws._ibis, self.name, column)

    def get_column_type(self, column: str) -> str | None:
        """Cached type for one column (alias of get_cached_type)."""
        return self.get_cached_type(column)

    def set_column_type(self, column: str, type_value: str) -> None:
        """Set a column's type to a refined categorical value (agent-facing).

        Only ``categorical_nominal`` / ``categorical_ordinal`` are accepted — the
        two refinements of the auto-detected ``categorical`` label. Continuous /
        datetime / identifier are auto-detected facts about the data and cannot be
        overridden here; to change those, transform the column. Raises ValueError
        for an unknown type or a column that does not exist.
        """
        from .validation.dtypes import SETTABLE_TYPES as _SETTABLE_TYPES
        if type_value not in _SETTABLE_TYPES:
            raise ValueError(
                f"set_column_type accepts only {sorted(_SETTABLE_TYPES)}; "
                f"got {type_value!r}. (continuous/datetime/identifier are auto-detected.)"
            )
        self._require_column(column)
        _ducktable.set_column_type(self._ws._ibis, self.name, column, type_value)

    def unset_column_type(self, column: str) -> None:
        """Clear a column's assigned type so it is re-detected on next use."""
        _ducktable.unset_column_type(self._ws._ibis, self.name, column)

    def column_types(self) -> dict[str, str]:
        """All cached types for this table as {column: type}."""
        return _ducktable.column_types(self._ws._ibis, self.name)

    def classify_categorical_as_nominal(self) -> list[str]:
        """Refine every still-unclassified categorical column to nominal.

        A batch default: walks the columns, and for each one whose type is the
        auto-detected coarse ``categorical`` label, sets it to
        ``categorical_nominal``. Returns the names of the columns it refined.
        Useful as a starting point (an agent defaults everything to nominal, then
        refines the genuinely-ordered ones to ordinal) and as the mock an LLM
        would perform in tests. Already-refined columns are left untouched.
        """
        from .validation.dtypes import classify_column as _classify
        refined: list[str] = []
        for col in self._table.schema().names:
            if _classify(col, self) == "categorical":
                _ducktable.set_column_type(self._ws._ibis, self.name, col, "categorical_nominal")
                refined.append(col)
        return refined

    def _require_column(self, column: str) -> None:
        if column not in self._table.schema().names:
            raise KeyError(
                f"No column {column!r} in table {self.name!r}. "
                f"Known: {list(self._table.schema().names)}"
            )

    def _invalidate_column_type_if_exists(self, name: str) -> None:
        """Clear the cached type when a column is about to be overwritten.

        Called from the write paths: overwriting a column changes its data, so any
        previously-cached type (auto or agent-set) may be stale. A genuinely new
        column has no cached entry, so the unset is a no-op for it.
        """
        if name in self._table.schema().names:
            _ducktable.unset_column_type(self._ws._ibis, self.name, name)

    # --- descriptive ------------------------------------------------------ #

    def profile(self) -> Result:
        return descriptive.profile(self)

    def detect_outliers(self, column: str) -> Result:
        return descriptive.detect_outliers(self, column)

    def association_matrix(self) -> Result:
        return descriptive.association_matrix(self)

    # --- feature computation (build new model-eligible columns) ----------- #

    def combine_columns(self, col_a: str, col_b: str, op: str, name: str | None = None) -> Result:
        return feature_computation.combine_columns(self, col_a, col_b, op, name)

    def transform_column(self, column: str, func: str, name: str | None = None) -> Result:
        return feature_computation.transform_column(self, column, func, name)

    def bin_column(
        self, column: str, n_bins: int = 4, strategy: str = "quantile", name: str | None = None
    ) -> Result:
        return feature_computation.bin_column(self, column, n_bins, strategy, name)

    def expand_datetime(self, column: str, parts: list[str] | None = None) -> Result:
        return feature_computation.expand_datetime(self, column, parts)

    def group_aggregate(
        self,
        group_by: str,
        value: str,
        agg: str = "mean",
        name: str | None = None,
        add_deviation: bool = False,
    ) -> Result:
        return feature_computation.group_aggregate(self, group_by, value, agg, name, add_deviation)

    def row_aggregate(self, columns: list[str], agg: str = "sum", name: str | None = None) -> Result:
        return feature_computation.row_aggregate(self, columns, agg, name)

    def normalize_fractions(self, columns: list[str], suffix: str = "_frac") -> Result:
        return feature_computation.normalize_fractions(self, columns, suffix)

    def compute_feature(self, name: str, expression: str) -> Result:
        return feature_computation.compute_feature(self, name, expression)

    # --- association ------------------------------------------------------ #

    def analyze_association(self, col_a: str, col_b: str) -> Result:
        return association.analyze_association(self, col_a, col_b)

    # --- clustering ------------------------------------------------------- #

    def cluster(self, n_clusters: int | None = None) -> Result:
        return clustering.cluster(self, n_clusters)

    def profile_clusters(self) -> Result:
        return clustering.profile_clusters(self)

    # --- supervised ------------------------------------------------------- #

    def train_classifier(self, target: str, name: str | None = None, backend: str = "gbt") -> Any:
        model = supervised.train_classifier(self, target, backend=backend)
        if isinstance(model, Result):  # honesty seam declined — nothing to register
            return model
        self.models[name or target] = model
        return model

    def train_regressor(self, target: str, name: str | None = None, backend: str = "gbt") -> Any:
        model = supervised.train_regressor(self, target, backend=backend)
        if isinstance(model, Result):  # honesty seam declined — nothing to register
            return model
        self.models[name or target] = model
        return model

    def evaluate(self, model_name: str) -> Result:
        return supervised.evaluate(self, self._model(model_name))

    def add_predictions(self, model_name: str, column_name: str | None = None) -> Result:
        model = self._model(model_name)
        frame = self.get_frame()
        preds = model.predict(frame[model._feature_names])
        col = column_name or f"{model_name}_pred"
        self.write_back_column(col, list(preds))
        return Result(
            method="add_predictions",
            summary=f"Wrote {len(preds)} predictions to column {col!r}",
            values={"column": col, "n": int(len(preds))},
            metadata={"model": model_name, "table": self.name},
        )

    # --- interpretation --------------------------------------------------- #

    def feature_importance(self, model_name: str) -> Result:
        return interpretation.feature_importance(self._model(model_name))

    def explain_prediction(self, model_name: str, row: Any) -> Result:
        return interpretation.explain_prediction(self._model(model_name), row)

    # --- dimensionality reduction ----------------------------------------- #

    def reduce_dimensions(self, method: str = "pca", n_components: int = 2) -> Result:
        return dimreduction.reduce_dimensions(self, method, n_components)

    # --- time series ------------------------------------------------------ #

    def decompose(self, time_column: str, value_column: str) -> Result:
        return timeseries.decompose(self, time_column, value_column)

    def forecast(self, time_column: str, value_column: str, horizon: int = 10) -> Result:
        return timeseries.forecast(self, time_column, value_column, horizon)

    def detect_changepoints(self, time_column: str, value_column: str, penalty: float = 10.0) -> Result:
        return timeseries.detect_changepoints(self, time_column, value_column, penalty)

    # --- insight primitives ----------------------------------------------- #

    def explain_metric(self, target: str, max_depth: int = 3) -> Result:
        return insights.explain_metric(self, target, max_depth)

    def market_basket(
        self,
        transaction_column: str,
        item_column: str,
        min_support: float = 0.01,
        min_confidence: float = 0.2,
        max_rules: int = 50,
    ) -> Result:
        return basket.market_basket(
            self, transaction_column, item_column, min_support, min_confidence, max_rules
        )

    def causal_effect(
        self, treatment: str, outcome: str, confounders: list[str] | None = None
    ) -> Result:
        return causal.causal_effect(self, treatment, outcome, confounders)

    def rfm(self, customer_column: str, date_column: str, monetary_column: str) -> Result:
        return cohort.rfm(self, customer_column, date_column, monetary_column)

    def retention_cohorts(self, customer_column: str, date_column: str) -> Result:
        return cohort.retention_cohorts(self, customer_column, date_column)

    def compare_periods(self, time_column: str, value_column: str, split: str | None = None) -> Result:
        return compare.compare_periods(self, time_column, value_column, split)

    # --- internals -------------------------------------------------------- #

    def _model(self, name: str) -> Any:
        if name not in self.models:
            raise KeyError(f"No model {name!r} on table {self.name!r}. Known: {list(self.models)}")
        return self.models[name]


def data_root(allowed_dir: str | Path | None = None) -> Path:
    """The one directory this engine may touch — TABULAR_BASE, or the cwd.

    The app spawns the engine with TABULAR_BASE pointing at its own analytics
    folder, which is also where every CSV it hands over lives, so one root covers
    session databases, ingested files and saved reports alike.
    """
    return Path(allowed_dir or os.environ.get("TABULAR_BASE") or ".").resolve()


def confine(con: Any, allowed_dir: str | Path | None = None) -> Path:
    """Shut a DuckDB instance in to ``data_root`` and throw away the key.

    DuckDB reaches the filesystem from inside an ordinary SELECT — read_csv_auto,
    glob, COPY TO, ATTACH — so agent-authored SQL is an arbitrary-file read unless
    the engine itself refuses. This is that refusal, at the engine rather than in
    a keyword filter over the SQL.

    The two settings seal each other and the order is what does it.
    ``allowed_directories`` is a carve-out that still applies once
    ``enable_external_access`` is false, so the allow-list has to be written
    first; turning external access off then freezes it, because DuckDB refuses to
    change the allow-list while access is disabled and refuses to re-enable
    access on a running database. SET, PRAGMA and RESET are all dead ends
    afterwards, which is why no ``lock_configuration`` is needed — and it must be
    avoided, since it would also block the ``SET timezone`` that ibis issues when
    a second connection opens the same file.

    Idempotent for that same reason: connections to one database file share the
    instance, so the second Workspace over a session finds it already confined.
    """
    root = data_root(allowed_dir)
    root.mkdir(parents=True, exist_ok=True)
    if _external_access_disabled(con):
        return root
    literal = str(root).replace("'", "''")
    con.execute(f"SET allowed_directories = ['{literal}']")
    con.execute("SET enable_external_access = false")
    return root


def _external_access_disabled(con: Any) -> bool:
    try:
        value = con.execute("SELECT current_setting('enable_external_access')").fetchone()
    except Exception:
        return False
    return bool(value) and str(value[0]).lower() in {"false", "0"}


class Workspace:
    """An in-memory DuckDB database holding one or more named tables.

    Each Workspace is independent: a fresh Session gets its own database, so
    models and derived (write-back) columns never leak between sessions.
    """

    def __init__(self, db_path: str | None = None, allowed_dir: str | Path | None = None) -> None:
        # db_path=None → in-memory (isolated library use); a path → persistent,
        # so tables and write-back columns survive across processes (CLI/MCP).
        self._ibis = ibis.duckdb.connect(db_path) if db_path else ibis.duckdb.connect()
        self._db_path = db_path
        self._allowed_dir = confine(self._ibis.con, allowed_dir)
        self._tables: dict[str, Table] = {}
        if db_path:
            self._reattach()

    @classmethod
    def create(
        cls,
        paths: list[str],
        db_path: str | None = None,
        allowed_dir: str | Path | None = None,
    ) -> "Workspace":
        """Build a workspace holding the given CSVs as tables.

        Args:
            paths: CSV files to load as tables.
            db_path: On-disk DuckDB file for a persistent workspace, or None for
                an in-memory one.
            allowed_dir: The only directory this database may touch (see confine).
        """
        resolved = [Path(p).resolve() for p in paths]
        if not resolved:
            raise ValueError("At least one CSV path is required.")
        ws = cls(db_path, allowed_dir=allowed_dir)
        for path in resolved:
            ws.add_csv(str(path))
        return ws

    def _reattach(self) -> None:
        """Rebuild Table handles from tables already present in a persisted DB."""
        names = set(self._ibis.list_tables())
        for view in names:
            if view.startswith("_tbl_"):
                continue  # internal table, not a public view
            if f"_tbl_{view}" in names:
                self._tables[view] = Table(self, view, f"_tbl_{view}", view)

    def close(self) -> None:
        """Close the underlying DuckDB connection (releases the file lock)."""
        try:
            self._ibis.con.close()
        except Exception:
            pass

    def add_csv(self, path: str, name: str | None = None) -> Table:
        """Load a CSV as a new table in this workspace and return its handle."""
        csv_path = Path(path).resolve()
        # Sanitize whether the name came from the caller or the filename stem, so
        # an explicit name can never introduce a reserved/spaced bare identifier.
        table_name = self._unique_name(_sanitize(name or csv_path.stem))
        internal = f"_tbl_{table_name}"
        _ducktable.load_csv(self._ibis, csv_path, internal, table_name)

        table = Table(self, table_name, internal, table_name)
        self._tables[table_name] = table
        return table

    def table(self, name: str) -> Table:
        """Return the handle for a named table."""
        if name not in self._tables:
            raise KeyError(f"No table {name!r}. Known: {self.table_names}")
        return self._tables[name]

    @property
    def table_names(self) -> list[str]:
        return list(self._tables)

    def run_sql(self, query: str) -> Any:
        """Run SQL against the workspace; every table is visible by its name."""
        return self._ibis.sql(query).execute()

    def relationships(self) -> Any:
        """Detect foreign-key relationships and return the connection graph."""
        from .relationships import detect_relationships
        return detect_relationships(self)

    def create_table(
        self,
        name: str,
        columns: list[tuple[str, str]] | None = None,
        select_sql: str | None = None,
    ) -> Table:
        """Create a new named table, either from a schema or from a SELECT.

        Two mutually exclusive modes:

        - ``columns``: a list of ``(column_name, sql_type)`` pairs defining an
          empty, well-structured table. This is the target when the source data
          is messy: define the clean schema here, then copy rows in with
          ``insert_into`` (one query or many). Types are validated against a
          fixed allowlist.
        - ``select_sql``: materialize an arbitrary query over the workspace's
          tables as a new table in a single shot (the escape hatch behind join).

        Either way the result is a normal table that every single-table analytic
        works on. Returns its handle.
        """
        if (columns is None) == (select_sql is None):
            raise ValueError("Pass exactly one of `columns` (empty schema) or `select_sql`.")
        table_name = self._unique_name(_sanitize(name))
        internal = f"_tbl_{table_name}"
        if columns is not None:
            typed = [(str(col), _validate_type(str(sql_type))) for col, sql_type in columns]
            if not typed:
                raise ValueError("`columns` must define at least one column.")
            _ducktable.create_empty(self._ibis, typed, internal, table_name)
        else:
            _ducktable.create_from_query(self._ibis, select_sql, internal, table_name)
        table = Table(self, table_name, internal, table_name)
        self._tables[table_name] = table
        return table

    def insert_into(self, name: str, source_sql: str) -> int:
        """Copy rows into an existing table from a SELECT/VALUES query.

        The partner of create_table's schema mode: point ``source_sql`` at the
        messy source (``SELECT trim(col), CAST(...) FROM raw WHERE ...``) or supply
        literal ``VALUES`` rows, and the columns map positionally to the target's
        own columns while ``_ti_row`` is assigned automatically. Call it repeatedly
        to build a table up incrementally. Returns the number of rows inserted.
        """
        table = self.table(name)  # validates the name exists
        return _ducktable.insert_select(self._ibis, table._internal, source_sql)

    def join(
        self,
        tables: list[str],
        name: str | None = None,
        how: str = "left",
    ) -> Table:
        """Join two or more tables along detected foreign keys into a new table.

        The join path is inferred from the foreign-key graph: tables are chained
        together on their FK↔PK columns. Overlapping column names are disambiguated
        by prefixing with the source table name. The result is a normal table, so
        every single-table operation (profile, cluster, train_*, …) runs on it.

        Args:
            tables: Names of the tables to join (2 or more).
            name: Name for the new table. Defaults to ``join_<t1>_<t2>_...``.
            how: SQL join type ("left", "inner").

        Returns:
            The Table handle for the joined result.
        """
        if len(tables) < 2:
            raise ValueError("join needs at least two tables.")
        missing = [t for t in tables if t not in self._tables]
        if missing:
            raise KeyError(f"Unknown tables: {missing}. Known: {self.table_names}")

        graph = self.relationships()
        from_sql = self._build_join_sql(tables, graph, how)
        projection = self._build_projection(tables)
        select_sql = f"SELECT {projection} FROM {from_sql}"
        return self.create_table(name or "join_" + "_".join(tables), select_sql=select_sql)

    def _build_join_sql(self, tables: list[str], graph: Any, how: str) -> str:
        """Chain tables together on FK↔PK edges, greedily from the first table."""
        how_sql = how.upper()
        joined = [tables[0]]
        from_sql = _ducktable.quote_ident(tables[0])
        remaining = tables[1:]

        while remaining:
            for candidate in list(remaining):
                edge = find_join_edge(graph, joined, candidate)
                if edge is None:
                    continue
                left_tbl, left_col, right_col = edge
                from_sql += (
                    f' {how_sql} JOIN "{candidate}" '
                    f'ON "{left_tbl}"."{left_col}" = "{candidate}"."{right_col}"'
                )
                joined.append(candidate)
                remaining.remove(candidate)
                break
            else:
                raise ValueError(
                    f"No foreign-key path connects {remaining} to {joined}. "
                    "Use create_table(name, select_sql=...) for a manual join."
                )
        return from_sql

    def _build_projection(self, tables: list[str]) -> str:
        """Qualified column list; duplicate names get a unique <table>_ prefix."""
        seen: set[str] = set()
        cols: list[str] = []
        for tbl in tables:
            for col in self.table(tbl)._table.schema().names:
                target = col if col not in seen else f"{tbl}_{col}"
                # Guard against the prefixed name itself already existing.
                while target in seen:
                    target = f"{tbl}_{target}"
                seen.add(target)
                cols.append(f'"{tbl}"."{col}" AS "{target}"')
        return ", ".join(cols)

    def _unique_name(self, base: str) -> str:
        if base not in self._tables:
            return base
        i = 2
        while f"{base}_{i}" in self._tables:
            i += 1
        return f"{base}_{i}"
