"""Session — thin state holder over a multi-table Workspace.

A Session owns a Workspace (one DuckDB database holding one or more tables) and a
job registry. It contains no algorithms: per-table analytics live on the Table
handle (``session.table("orders").profile()``), and Session simply provides
access plus a few workspace-level conveniences (relationships, run_sql).

For the common single-table case, Session also forwards the analytics methods to
the sole table, so ``session.profile()`` still works without naming a table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .service.jobs.registry import JobRegistry
from ..shared.results import Result
from .service.workspace import Table, Workspace


class Session:
    """State holder for a single- or multi-table analysis session.

    Construct via ``Session.load(path)`` or ``Session.load([path, ...])``.
    """

    def __init__(self) -> None:
        self.workspace: Workspace | None = None
        self._jobs: JobRegistry = JobRegistry()
        self.id: str | None = None          # set for persistent (CLI/MCP) sessions
        self._dir: Path | None = None       # on-disk session directory, if any

    @classmethod
    def load(cls, paths: str | Path | list[str | Path]) -> "Session":
        """Load one or more CSVs into a new in-memory Session.

        This is the library entry point (no persistence). For addressable,
        on-disk sessions the CLI/MCP layers use tabint.analysis.db.persistence instead.

        Args:
            paths: A single CSV path, or a list of paths to load as related tables.

        Returns:
            A new Session backed by an in-memory Workspace holding the table(s).
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        session = cls()
        session.workspace = Workspace.create([str(p) for p in paths])
        return session

    @classmethod
    def _from_workspace(
        cls, workspace: Workspace, session_id: str | None = None, session_dir: Path | None = None
    ) -> "Session":
        """Wrap an existing Workspace (used by the persistence layer)."""
        session = cls()
        session.workspace = workspace
        session.id = session_id
        session._dir = session_dir
        return session

    def close(self) -> None:
        """Close the underlying workspace connection (releases the DB file)."""
        if self.workspace is not None:
            self.workspace.close()

    # --- workspace-level API ---------------------------------------------- #

    def add_table(self, path: str, name: str | None = None) -> Table:
        """Load another CSV into this session as a new table; return its handle."""
        return self.workspace.add_csv(path, name)

    def table(self, name: str) -> Table:
        """Return the handle for a named table (carries the analytics API)."""
        return self.workspace.table(name)

    @property
    def tables(self) -> list[str]:
        """Names of all tables in the session."""
        return self.workspace.table_names

    def relationships(self) -> Any:
        """Detect foreign-key relationships across the session's tables."""
        return self.workspace.relationships()

    def join(self, tables: list[str], name: str | None = None, how: str = "left") -> Table:
        """Join tables along detected foreign keys into a new table; return its handle.

        The result is a normal table, so every single-table operation runs on it:
        ``session.join(["orders", "customers"]).analyze_association(...)``.
        """
        return self.workspace.join(tables, name=name, how=how)

    def create_table(
        self,
        name: str,
        columns: list[tuple[str, str]] | None = None,
        select_sql: str | None = None,
    ) -> Table:
        """Create a new table from a column schema or from a SELECT.

        Pass ``columns`` (``(name, type)`` pairs) to define an empty structured
        table to copy messy data into with ``insert_into``, or ``select_sql`` to
        materialize a query in one shot.
        """
        return self.workspace.create_table(name, columns=columns, select_sql=select_sql)

    def insert_into(self, name: str, source_sql: str) -> int:
        """Copy rows into an existing table from a SELECT/VALUES query; return the count."""
        return self.workspace.insert_into(name, source_sql)

    def run_sql(self, query: str) -> Any:
        """Run SQL across the workspace; every table is visible by its name."""
        return self.workspace.run_sql(query)

    # --- single-table convenience (delegates to the sole table) ----------- #

    def count_rows(self) -> int:
        return self._sole().count_rows()

    def count_non_null(self, column: str) -> int:
        return self._sole().count_non_null(column)

    def profile(self) -> Result:
        return self._sole().profile()

    def detect_outliers(self, column: str) -> Result:
        return self._sole().detect_outliers(column)

    def association_matrix(self) -> Result:
        return self._sole().association_matrix()

    def combine_columns(self, col_a: str, col_b: str, op: str, name: str | None = None) -> Result:
        return self._sole().combine_columns(col_a, col_b, op, name)

    def transform_column(self, column: str, func: str, name: str | None = None) -> Result:
        return self._sole().transform_column(column, func, name)

    def bin_column(
        self, column: str, n_bins: int = 4, strategy: str = "quantile", name: str | None = None
    ) -> Result:
        return self._sole().bin_column(column, n_bins, strategy, name)

    def expand_datetime(self, column: str, parts: list[str] | None = None) -> Result:
        return self._sole().expand_datetime(column, parts)

    def group_aggregate(
        self,
        group_by: str,
        value: str,
        agg: str = "mean",
        name: str | None = None,
        add_deviation: bool = False,
    ) -> Result:
        return self._sole().group_aggregate(group_by, value, agg, name, add_deviation)

    def row_aggregate(self, columns: list[str], agg: str = "sum", name: str | None = None) -> Result:
        return self._sole().row_aggregate(columns, agg, name)

    def normalize_fractions(self, columns: list[str], suffix: str = "_frac") -> Result:
        return self._sole().normalize_fractions(columns, suffix)

    def compute_feature(self, name: str, expression: str) -> Result:
        return self._sole().compute_feature(name, expression)

    def analyze_association(self, col_a: str, col_b: str) -> Result:
        return self._sole().analyze_association(col_a, col_b)

    def cluster(self, n_clusters: int | None = None) -> Result:
        return self._sole().cluster(n_clusters)

    def profile_clusters(self) -> Result:
        return self._sole().profile_clusters()

    def train_classifier(self, target: str, name: str | None = None, backend: str = "gbt") -> Any:
        return self._sole().train_classifier(target, name, backend=backend)

    def train_regressor(self, target: str, name: str | None = None, backend: str = "gbt") -> Any:
        return self._sole().train_regressor(target, name, backend=backend)

    def evaluate(self, model_name: str) -> Result:
        return self._sole().evaluate(model_name)

    def add_predictions(self, model_name: str, column_name: str | None = None) -> Result:
        return self._sole().add_predictions(model_name, column_name)

    def feature_importance(self, model_name: str) -> Result:
        return self._sole().feature_importance(model_name)

    def explain_prediction(self, model_name: str, row: Any) -> Result:
        return self._sole().explain_prediction(model_name, row)

    def reduce_dimensions(self, method: str = "pca", n_components: int = 2) -> Result:
        return self._sole().reduce_dimensions(method, n_components)

    def decompose(self, time_column: str, value_column: str) -> Result:
        return self._sole().decompose(time_column, value_column)

    def forecast(self, time_column: str, value_column: str, horizon: int = 10) -> Result:
        return self._sole().forecast(time_column, value_column, horizon)

    def detect_changepoints(self, time_column: str, value_column: str, penalty: float = 10.0) -> Result:
        return self._sole().detect_changepoints(time_column, value_column, penalty)

    def explain_metric(self, target: str, max_depth: int = 3) -> Result:
        return self._sole().explain_metric(target, max_depth)

    def market_basket(
        self,
        transaction_column: str,
        item_column: str,
        min_support: float = 0.01,
        min_confidence: float = 0.2,
        max_rules: int = 50,
    ) -> Result:
        return self._sole().market_basket(
            transaction_column, item_column, min_support, min_confidence, max_rules
        )

    def causal_effect(
        self, treatment: str, outcome: str, confounders: list[str] | None = None
    ) -> Result:
        return self._sole().causal_effect(treatment, outcome, confounders)

    def rfm(self, customer_column: str, date_column: str, monetary_column: str) -> Result:
        return self._sole().rfm(customer_column, date_column, monetary_column)

    def retention_cohorts(self, customer_column: str, date_column: str) -> Result:
        return self._sole().retention_cohorts(customer_column, date_column)

    def compare_periods(self, time_column: str, value_column: str, split: str | None = None) -> Result:
        return self._sole().compare_periods(time_column, value_column, split)

    @property
    def models(self) -> dict[str, Any]:
        """Model registry of the sole table (single-table sessions)."""
        return self._sole().models

    # --- column-type classification (forwarded to the sole table) -------- #

    def classify_categorical_as_nominal(self) -> list[str]:
        """Refine every unclassified categorical column on the sole table to nominal."""
        return self._sole().classify_categorical_as_nominal()

    def set_column_type(self, column: str, type: str) -> None:
        """Set a column's type on the sole table (single-table sessions)."""
        self._sole().set_column_type(column, type)

    @property
    def _store(self) -> Table | None:
        """The sole table, or None — kept for backward compatibility."""
        if self.workspace is None or len(self.workspace.table_names) != 1:
            return None
        return self._sole()

    # --- internals -------------------------------------------------------- #

    def _sole(self) -> Table:
        names = self.workspace.table_names if self.workspace else []
        if len(names) == 1:
            return self.workspace.table(names[0])
        raise ValueError(
            f"This session has {len(names)} tables {names}; call "
            f"session.table(<name>).<method>() instead of the session shortcut."
        )
