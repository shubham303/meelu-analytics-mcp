"""Analysis MCP tools — the ~30 data-analysis tools plus session lifecycle,
structure (relationships/join/sql/create_table/insert_into).

Ported into Meelu from TableIntelligence (same author): the outreach and
platform-integration modules (Stripe connector, entitlement) were deliberately
left behind — external data arrives through Meelu's own connectors, and Meelu's
backend is the report store. This engine computes; it does not publish.

These tools operate on the Session facade (analysis.session) and the live session
registry held in shared.server. They register onto the single FastMCP instance.
"""
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from tabint.analysis.db import persistence
from tabint.analysis.service.validation.dtypes import classify_column, SETTABLE_TYPES
from tabint.analysis.service.workspace import data_root
from tabint.shared.serialize import jsonable as _jsonable, result_dict as _result
from tabint.shared.results import Result
from tabint.shared import server
from tabint.shared.server import mcp, get_session as _get, register_session


def _BASE() -> str:
    """Live session base root — read from ``shared.server`` so tests that re-point
    the registry (``server._BASE = tmp``) take effect for every tool, not just the
    ones that reopened via ``get_session``."""
    return server._BASE

def _ingest_paths(paths: list[str]):
    """Resolve ingest paths, staging anything that lives outside the data root.

    Loading a CSV is a one-way trip: ``read_csv_auto`` copies the rows into
    DuckDB and nothing afterwards refers back to the file. So *where the file
    sat* is irrelevant to every question the session can answer — making users
    move a CSV into a blessed folder before naming it bought no safety, only
    friction.

    What the data root does still buy is the seal in ``workspace.confine``:
    ``enable_external_access = false`` stops agent-authored SQL (``run_sql``,
    ``create_table``, ``insert_into``) from reading arbitrary files behind a
    plain-looking SELECT. That seal is the boundary worth keeping, and widening
    the allow-list per ingest would break it permanently for the session.

    So an outside path is copied into a unique directory under ``<root>/.staging``
    and read from there, and the copy is deleted once the rows are in DuckDB. The
    unique part is the directory, not the filename, because the table takes its
    name from the file's stem — ``~/Downloads/orders.csv`` must still land as
    ``orders``. The database's view of the filesystem never changes; only this
    function, acting on a path the caller named explicitly, reaches out.

    Returns a context manager yielding resolved paths, or an error dict.
    """
    root = data_root(_BASE())
    missing = [str(Path(p).expanduser()) for p in paths
               if not Path(p).expanduser().resolve().is_file()]
    if missing:
        return {"ok": False, "error": "file_not_found",
                "message": f"No readable file at: {missing}. Pass the path to a CSV file — "
                           f"any location this process can read is fine."}
    return _staged(root, [Path(p).expanduser().resolve() for p in paths])


@contextmanager
def _staged(root: Path, resolved: list[Path]):
    """Yield readable-by-the-engine paths, cleaning up any staged copies after."""
    staging = root / ".staging"
    copies: list[Path] = []  # staging directories to remove afterwards
    out: list[str] = []
    try:
        for full in resolved:
            if full == root or root in full.parents:
                out.append(str(full))
                continue
            holder = staging / uuid4().hex
            holder.mkdir(parents=True, exist_ok=True)
            dest = holder / full.name
            shutil.copyfile(full, dest)
            copies.append(holder)
            out.append(str(dest))
        yield out
    finally:
        for holder in copies:
            shutil.rmtree(holder, ignore_errors=True)


def _summary(session) -> dict:
    return {
        "session_key": session.id,
        "tables": session.tables,
        "relationships": _jsonable(session.relationships().model_dump()),
    }


@mcp.tool()
def create_session(paths: list[str]) -> dict:
    """Create a session from one or more CSV paths. Returns the session_key,
    the loaded table names, and the auto-detected foreign-key relationships.
    A path may point anywhere this process can read — the rows are copied into
    the session's database on load, so the file itself is not consulted again."""
    staged = _ingest_paths(paths)
    if isinstance(staged, dict):
        return staged
    with staged as safe:
        session = persistence.create_session(safe, base=_BASE())
    register_session(session)
    return _summary(session)


@mcp.tool()
def list_sessions() -> list[str]:
    """List the keys of all persisted sessions."""
    return persistence.list_sessions(base=_BASE())




@mcp.tool()
def session_info(session_key: str) -> dict:
    """Return a session's tables and detected relationships."""
    return _summary(_get(session_key))


@mcp.tool()
def add_table(session_key: str, path: str) -> dict:
    """Load another CSV into an existing session as a new table. The path may point
    anywhere this process can read — the rows are copied into the session's
    database on load, so the file itself is not consulted again."""
    session = _get(session_key)
    staged = _ingest_paths([path])
    if isinstance(staged, dict):
        return staged
    with staged as safe:
        table = session.add_table(safe[0])
    return {"session_key": session_key, "added_table": table.name, "tables": session.tables}


@mcp.tool()
def relationships(session_key: str) -> dict:
    """Detect and return the foreign-key graph across the session's tables."""
    return _jsonable(_get(session_key).relationships().model_dump())


@mcp.tool()
def join(session_key: str, tables: list[str], name: str | None = None, how: str = "left") -> dict:
    """Join tables along detected foreign keys into a new table; returns its name and columns."""
    joined = _get(session_key).join(tables, name=name, how=how)
    frame = joined.get_frame()
    return {"table": joined.name, "columns": list(frame.columns), "n_rows": int(len(frame))}


@mcp.tool()
def run_sql(session_key: str, query: str, limit: int = 1000) -> dict:
    """Run a read-only SQL SELECT across the session's tables (each visible by name).
    Rows are capped at `limit`. To build or fill tables, use create_table / insert_into."""
    frame = _get(session_key).run_sql(query)
    total = int(len(frame))
    records = _jsonable(frame.head(limit).to_dict(orient="records"))
    return {"n_rows": total, "truncated": total > limit, "rows": records}


@mcp.tool()
def create_table(
    session_key: str,
    name: str,
    columns: list[dict] | None = None,
    select_sql: str | None = None,
) -> dict:
    """Create a new clean, structured table in the session.

    Use this when the source data is messy or badly shaped: define the correct
    schema here, then copy the data across with insert_into (one query at a time
    or in bulk). run_sql cannot create tables — this is the tool that does.

    Two mutually exclusive modes (pass exactly one):
    - columns: an empty typed table. Each entry is {"name": "...", "type": "..."},
      e.g. [{"name": "order_id", "type": "BIGINT"}, {"name": "amount", "type": "DECIMAL(10,2)"}].
      Allowed types are the standard SQL/DuckDB types (INTEGER, BIGINT, DOUBLE,
      DECIMAL(p,s), VARCHAR, DATE, TIMESTAMP, BOOLEAN, ...).
    - select_sql: materialize a query over the existing tables as a new table in
      one shot (e.g. "SELECT trim(name) AS name, CAST(qty AS INTEGER) AS qty FROM raw").

    Returns the created table's name and columns.
    """
    cols = None
    if columns is not None:
        cols = [(c["name"], c["type"]) for c in columns]
    table = _get(session_key).create_table(name, columns=cols, select_sql=select_sql)
    frame = table.get_frame()
    return {"table": table.name, "columns": list(frame.columns), "n_rows": int(len(frame))}


@mcp.tool()
def insert_into(session_key: str, name: str, source_sql: str) -> dict:
    """Copy rows into an existing table (partner of create_table's `columns` mode).

    `source_sql` is a SELECT or VALUES query whose columns map positionally to the
    target table's columns, e.g.
      "SELECT trim(customer) AS name, CAST(spend AS DECIMAL(10,2)) FROM raw WHERE spend IS NOT NULL"
    or "VALUES ('Acme', 12.50), ('Globex', 9.99)".
    Call repeatedly to build a table up from many messy sources. Returns the
    number of rows inserted and the table's new total row count.
    """
    session = _get(session_key)
    inserted = session.insert_into(name, source_sql)
    total = int(len(session.table(name).get_frame()))
    return {"table": name, "inserted": inserted, "n_rows": total}




@mcp.tool()
def count_rows(session_key: str, table: str) -> dict:
    """Number of rows in a table — a cheap in-database COUNT(*), no data materialized.

    Use this instead of `profile` when you only need the row count (e.g. to size a
    table before an operation); it stays fast on arbitrarily large tables.
    """
    return {"table": table, "n_rows": _get(session_key).table(table).count_rows()}


@mcp.tool()
def count_non_null(session_key: str, table: str, column: str) -> dict:
    """Number of non-NULL (non-NaN) values in a column — an in-database COUNT(col).

    Returns the non-null count plus the row total and derived null count, all from
    a cheap COUNT with no data materialized. Fast on arbitrarily large tables.
    """
    t = _get(session_key).table(table)
    n_rows = t.count_rows()
    n_non_null = t.count_non_null(column)
    return {
        "table": table,
        "column": column,
        "n_non_null": n_non_null,
        "n_rows": n_rows,
        "n_null": n_rows - n_non_null,
    }


@mcp.tool()
def list_categorical_columns(session_key: str, table: str) -> dict:
    """Return the categorical columns that still need a type assignment.

    A worklist, not a full listing: a column appears here only while its type is
    still the auto-detected coarse label ``categorical``. Once you refine it with
    set_column_type (to categorical_nominal or categorical_ordinal) it drops off
    this list. An empty ``unclassified`` means every categorical column is typed
    and the table is ready for modeling. Each entry includes the distinct value
    count and a small sample of the values to help you judge whether the levels
    are ordered (ordinal) or not (nominal).
    """
    t = _get(session_key).table(table)
    frame = t.get_frame()
    out = []
    for col in t._table.schema().names:
        # classify_column is cache-and-compute: this call also caches the type,
        # so every column ends up typed after a pass through this tool.
        if classify_column(col, t) == "categorical":
            series = frame[col].dropna()
            distinct = int(series.nunique())
            sample = [str(v) for v in series.unique()[:10]]
            out.append({"column": col, "distinct": distinct, "sample": sample})
    return {"table": table, "unclassified": out, "n": len(out)}


@mcp.tool()
def set_column_type(session_key: str, table: str, column: str, type: str) -> dict:
    """Assign a refined categorical type to a column.

    The only accepted values are ``categorical_nominal`` (unordered categories:
    color, city, payment method) and ``categorical_ordinal`` (ordered categories
    with no meaningful spacing: satisfaction, education level, size S/M/L).
    continuous / datetime / identifier are auto-detected facts about the data and
    cannot be set here. The assignment persists with the session and feeds every
    later operation: ordinal columns are integer-encoded at modeling time rather
    than one-hot encoded, and modeling refuses to run until every categorical
    column is refined.
    """
    if type not in SETTABLE_TYPES:
        raise ValueError(
            f"set_column_type accepts only {sorted(SETTABLE_TYPES)}; got {type!r}. "
            "(continuous/datetime/identifier are auto-detected, not settable.)"
        )
    t = _get(session_key).table(table)
    t.set_column_type(column, type)
    return {"table": table, "column": column, "type": type}


@mcp.tool()
def unset_column_type(session_key: str, table: str, column: str) -> dict:
    """Clear a column's assigned type so it is re-detected on next use.

    Useful to correct a mistake: after unsetting, the column returns to the
    auto-detected coarse label and re-appears on the list_categorical_columns
    worklist. The next classify_column call recomputes and re-caches its type.
    """
    t = _get(session_key).table(table)
    t.unset_column_type(column)
    return {"table": table, "column": column, "cleared": True}


@mcp.tool()
def classify_as_nominal(session_key: str, table: str) -> dict:
    """Refine every unclassified categorical column on the table to nominal.

    A batch convenience over set_column_type: assigns categorical_nominal to
    every column still at the coarse ``categorical`` label in one call. Typical
    workflow — call this first to unblock modeling, then selectively refine the
    genuinely-ordered columns (satisfaction, size, rating) to categorical_ordinal
    with set_column_type. Returns the list of columns it refined.
    """
    t = _get(session_key).table(table)
    refined = t.classify_categorical_as_nominal()
    return {"table": table, "refined": refined, "n": len(refined)}


@mcp.tool()
def profile(session_key: str, table: str) -> dict:
    """Profile every column of a table: type, missingness, cardinality, distribution."""
    return _result(_get(session_key).table(table).profile())


@mcp.tool()
def detect_outliers(session_key: str, table: str, column: str) -> dict:
    """Flag outliers in a numeric column (IQR + z-score) and write the flags back as a column."""
    return _result(_get(session_key).table(table).detect_outliers(column))


@mcp.tool()
def analyze_association(session_key: str, table: str, col_a: str, col_b: str) -> dict:
    """Test the association between two columns; the test is chosen from the dtype pair."""
    return _result(_get(session_key).table(table).analyze_association(col_a, col_b))


@mcp.tool()
def association_matrix(session_key: str, table: str) -> dict:
    """Pairwise association strength across all column pairs of a table."""
    return _result(_get(session_key).table(table).association_matrix())


# --------------------------------------------------------------------------- #
# feature computation: build new model-eligible columns from existing ones
# --------------------------------------------------------------------------- #

@mcp.tool()
def combine_columns(
    session_key: str, table: str, col_a: str, col_b: str, op: str, name: str | None = None
) -> dict:
    """Create a feature by combining two numeric columns with an arithmetic op.

    `op` is one of: add, subtract, multiply, divide, ratio. Division-by-zero
    becomes NaN. This is the primitive for most domain features — e.g.
    density = mass / volume: you supply the columns and the op, the arithmetic is
    generic. The new column is written back and is eligible for modelling.
    """
    return _result(_get(session_key).table(table).combine_columns(col_a, col_b, op, name))


@mcp.tool()
def transform_column(
    session_key: str, table: str, column: str, func: str, name: str | None = None
) -> dict:
    """Create a feature by applying a math transform to one numeric column.

    `func` is one of: log, log1p, sqrt, square, reciprocal, abs, zscore. Values
    outside a transform's domain (e.g. log of a non-positive) become NaN. Use log
    to tame skew, zscore to standardise, etc.
    """
    return _result(_get(session_key).table(table).transform_column(column, func, name))


@mcp.tool()
def bin_column(
    session_key: str,
    table: str,
    column: str,
    n_bins: int = 4,
    strategy: str = "quantile",
    name: str | None = None,
) -> dict:
    """Discretise a numeric column into ordinal bins (a categorical feature).

    `strategy` = "quantile" (equal-frequency) or "uniform" (equal-width). The new
    column holds 0-based integer bin indices.
    """
    return _result(_get(session_key).table(table).bin_column(column, n_bins, strategy, name))


@mcp.tool()
def expand_datetime(
    session_key: str, table: str, column: str, parts: list[str] | None = None
) -> dict:
    """Expand a datetime column into calendar-component features.

    `parts` (default: year, month, dayofweek, is_weekend) is any subset of: year,
    quarter, month, week, day, dayofweek, dayofyear, hour, is_weekend,
    is_month_start, is_month_end. Each becomes `<column>_<part>`.
    """
    return _result(_get(session_key).table(table).expand_datetime(column, parts))


@mcp.tool()
def group_aggregate(
    session_key: str,
    table: str,
    group_by: str,
    value: str,
    agg: str = "mean",
    name: str | None = None,
    add_deviation: bool = False,
) -> dict:
    """Aggregate `value` within each `group_by` category, broadcast back to rows.

    Every row receives its group's statistic (e.g. each order gets its customer's
    mean spend) — a strong relational feature. `agg` is one of mean, sum, min,
    max, std, median, count. With `add_deviation=True`, also writes
    `<value>_dev_from_<group_by>` = value − group mean.
    """
    return _result(
        _get(session_key).table(table).group_aggregate(group_by, value, agg, name, add_deviation)
    )


@mcp.tool()
def row_aggregate(
    session_key: str, table: str, columns: list[str], agg: str = "sum", name: str | None = None
) -> dict:
    """Aggregate several numeric columns across each row into one feature.

    `agg` is one of mean, sum, min, max, std, median, count (count = number of
    non-null inputs). The generic form of a total like "total atom count" from
    per-element count columns.
    """
    return _result(_get(session_key).table(table).row_aggregate(columns, agg, name))


@mcp.tool()
def normalize_fractions(
    session_key: str, table: str, columns: list[str], suffix: str = "_frac"
) -> dict:
    """Turn a set of count/amount columns into per-row fractions of their total.

    Each `<col>` becomes `<col><suffix>` = col / (row sum across the set), so the
    new columns sum to 1 per row. The generic form of composition fractions.
    """
    return _result(_get(session_key).table(table).normalize_fractions(columns, suffix))


@mcp.tool()
def compute_feature(session_key: str, table: str, name: str, expression: str) -> dict:
    """Create one feature column from a custom SQL scalar expression — the escape
    hatch when the fixed feature tools can't express what you need.

    `expression` is a DuckDB scalar expression over the table's columns, evaluated
    per row INSIDE the database (nothing is streamed to the app, so it scales to
    massive tables), and stored as a new model-eligible column `name`. Examples:
      - "mass / NULLIF(volume, 0)"
      - "CASE WHEN age >= 18 THEN 'adult' ELSE 'minor' END"
      - "avg(spend) OVER (PARTITION BY customer_id)"
      - "regexp_extract(email, '@(.*)$', 1)"

    Strictly feature generation: it must be a single scalar expression. Statement
    chaining, subqueries, DDL/DML, and file/catalog functions (read_csv, attach,
    install, ...) are rejected, and the expression must reference existing columns.
    """
    return _result(_get(session_key).table(table).compute_feature(name, expression))


# --------------------------------------------------------------------------- #
# clustering / dimensionality reduction
# --------------------------------------------------------------------------- #

@mcp.tool()
def cluster(session_key: str, table: str, n_clusters: int | None = None) -> dict:
    """Cluster rows (k-means; k auto-selected by silhouette if omitted) and write labels back."""
    return _result(_get(session_key).table(table).cluster(n_clusters))


@mcp.tool()
def profile_clusters(session_key: str, table: str) -> dict:
    """Characterize each cluster (requires cluster() to have been run first)."""
    return _result(_get(session_key).table(table).profile_clusters())


@mcp.tool()
def reduce_dimensions(session_key: str, table: str, method: str = "pca", n_components: int = 2) -> dict:
    """Reduce a table to a few components (pca/tsne/umap) and write them back as columns."""
    return _result(_get(session_key).table(table).reduce_dimensions(method, n_components))


# --------------------------------------------------------------------------- #
# supervised + interpretation
# --------------------------------------------------------------------------- #

@mcp.tool()
def train_classifier(
    session_key: str, table: str, target: str, name: str | None = None, backend: str = "gbt"
) -> dict:
    """Train a classifier on a table and persist it under `name` (default: target).

    backend: "gbt" (default gradient-boosted trees) or "tabicl" (TabICL v2
    foundation model — no per-task training, strong on small/medium tables,
    needs the optional `tabicl` dependency).
    """
    return _train(session_key, table, target, name, "classification", backend)


@mcp.tool()
def train_regressor(
    session_key: str, table: str, target: str, name: str | None = None, backend: str = "gbt"
) -> dict:
    """Train a regressor on a table and persist it under `name` (default: target).

    backend: "gbt" (default gradient-boosted trees) or "tabicl" (TabICL v2
    foundation model — needs the optional `tabicl` dependency).
    """
    return _train(session_key, table, target, name, "regression", backend)


def _train(
    session_key: str, table: str, target: str, name: str | None, task: str, backend: str = "gbt"
) -> dict:
    session = _get(session_key)
    handle = session.table(table)
    model_name = name or target
    if task == "classification":
        model = handle.train_classifier(target, name=model_name, backend=backend)
    else:
        model = handle.train_regressor(target, name=model_name, backend=backend)
    if isinstance(model, Result):  # honesty seam declined training — surface it, don't save
        return _result(model)
    persistence.save_model(session, table, model_name, model)
    return {"model_name": model_name, "table": table, "target": target, "task": task,
            "backend": backend, "features": model._feature_names}


@mcp.tool()
def evaluate(session_key: str, table: str, model_name: str) -> dict:
    """Evaluate a trained model on its held-out test split."""
    return _result(_get(session_key).table(table).evaluate(model_name))


@mcp.tool()
def feature_importance(session_key: str, table: str, model_name: str) -> dict:
    """Permutation feature importance for a trained model."""
    return _result(_get(session_key).table(table).feature_importance(model_name))


@mcp.tool()
def add_predictions(session_key: str, table: str, model_name: str, column_name: str | None = None) -> dict:
    """Write a trained model's predictions back onto the table as a new column."""
    return _result(_get(session_key).table(table).add_predictions(model_name, column_name))


@mcp.tool()
def explain_prediction(session_key: str, table: str, model_name: str, row_index: int = 0) -> dict:
    """Explain a single prediction with SHAP; row_index is the 0-based table row."""
    handle = _get(session_key).table(table)
    row = handle.get_frame().iloc[int(row_index)].to_dict()
    return _result(handle.explain_prediction(model_name, row))


# --------------------------------------------------------------------------- #
# time series
# --------------------------------------------------------------------------- #

@mcp.tool()
def decompose(session_key: str, table: str, time_column: str, value_column: str) -> dict:
    """Decompose a time series into trend / seasonality / residual."""
    return _result(_get(session_key).table(table).decompose(time_column, value_column))


@mcp.tool()
def forecast(session_key: str, table: str, time_column: str, value_column: str, horizon: int = 10) -> dict:
    """Forecast a time series forward `horizon` steps (ARIMA)."""
    return _result(_get(session_key).table(table).forecast(time_column, value_column, horizon))


@mcp.tool()
def detect_changepoints(
    session_key: str, table: str, time_column: str, value_column: str, penalty: float = 10.0
) -> dict:
    """Detect points where a time series shifts behaviour (ruptures PELT).

    Needs the optional `insights` extra. Higher `penalty` = fewer changepoints.
    """
    return _result(_get(session_key).table(table).detect_changepoints(time_column, value_column, penalty))


# --------------------------------------------------------------------------- #
# insight primitives
# --------------------------------------------------------------------------- #

@mcp.tool()
def explain_metric(session_key: str, table: str, target: str, max_depth: int = 3) -> dict:
    """Explain a metric: ranked drivers + interpretable segment rules (shallow tree)."""
    return _result(_get(session_key).table(table).explain_metric(target, max_depth))


@mcp.tool()
def market_basket(
    session_key: str,
    table: str,
    transaction_column: str,
    item_column: str,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    max_rules: int = 50,
) -> dict:
    """Association-rule mining ("buy X → also buy Y"). Needs the optional `insights` extra."""
    return _result(_get(session_key).table(table).market_basket(
        transaction_column, item_column, min_support, min_confidence, max_rules))


@mcp.tool()
def causal_effect(
    session_key: str,
    table: str,
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
) -> dict:
    """Estimate the causal effect of `treatment` on `outcome` (DoWhy backdoor).

    Needs the optional `insights` extra. Defaults confounders to all other features.
    """
    return _result(_get(session_key).table(table).causal_effect(treatment, outcome, confounders))


@mcp.tool()
def rfm(session_key: str, table: str, customer_column: str, date_column: str, monetary_column: str) -> dict:
    """RFM quintile segmentation of customers (Champions, At Risk, ...)."""
    return _result(_get(session_key).table(table).rfm(customer_column, date_column, monetary_column))


@mcp.tool()
def retention_cohorts(session_key: str, table: str, customer_column: str, date_column: str) -> dict:
    """Monthly retention matrix: first-purchase cohort × months-since."""
    return _result(_get(session_key).table(table).retention_cohorts(customer_column, date_column))


@mcp.tool()
def compare_periods(
    session_key: str, table: str, time_column: str, value_column: str, split: str | None = None
) -> dict:
    """Compare a metric before vs after a cut date (means, % change, significance)."""
    return _result(_get(session_key).table(table).compare_periods(time_column, value_column, split))


