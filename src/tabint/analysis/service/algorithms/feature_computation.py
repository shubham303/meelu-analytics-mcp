"""Generic feature-computation family — build new columns from existing ones.

Unlike the rest of analytics (which *reads* the given columns), this module
*creates* columns: the domain-agnostic feature-engineering primitives an agent
composes to turn raw fields into model-ready features. Every function writes its
result back with ``feature=True`` so the new columns are eligible in feature
matrices (via feature_columns) rather than being treated as derived annotations.

Scope is deliberately generic — arithmetic combinations, math transforms,
binning, datetime expansion, group/row aggregates, and count→fraction
normalisation. Domain features (e.g. a materials-science lattice volume) are just
the arithmetic primitive applied with domain knowledge the *agent* supplies; we
provide the mechanism, not the chemistry.

Library note: these are column transforms, not statistical algorithms, so pandas
/ numpy express them directly and clearly — there is no proven library to wrap
the way association/causal primitives wrap scipy/DoWhy.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ....shared import honesty
from ....shared.results import Result


def _deterministic_trust(*extra_caveats: str) -> honesty.Trust:
    """Trust for a deterministic column transform — no inferential claim, so HIGH,
    but flag that the transform must match intent (and note any coercion/drops)."""
    return honesty.Trust(
        level=honesty.TrustLevel.HIGH,
        caveats=[
            "This is a deterministic transformation, not a statistical estimate — "
            "there's no inference uncertainty, but check the transform matches your intent.",
            *extra_caveats,
        ],
        basis=["deterministic transform"],
    )


# --------------------------------------------------------------------------- #
# operation vocabularies (explicit maps — never eval arbitrary expressions)
# --------------------------------------------------------------------------- #

# Binary column-to-column operations. Division guards divide-by-zero by turning
# ±inf into NaN so a zero denominator becomes missing, not a poisoned feature.
_BINARY_OPS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "ratio": lambda a, b: a / b,  # alias of divide, reads better as a feature
}

# Unary math transforms. log/sqrt are only defined on part of the domain; values
# outside it become NaN rather than raising, so one bad row can't sink a column.
_UNARY_FUNCS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "log": lambda s: np.log(s.where(s > 0)),
    "log1p": lambda s: np.log1p(s.where(s > -1)),
    "sqrt": lambda s: np.sqrt(s.where(s >= 0)),
    "square": lambda s: s ** 2,
    "reciprocal": lambda s: 1.0 / s,
    "abs": lambda s: s.abs(),
    "zscore": lambda s: (s - s.mean()) / s.std(ddof=0),
}

# Aggregations usable both across rows-within-a-group and across columns-in-a-row.
_AGGS = {"mean", "sum", "min", "max", "std", "median", "count"}

# Standard components pulled out of a datetime column.
_DATETIME_PARTS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "year": lambda dt: dt.dt.year,
    "quarter": lambda dt: dt.dt.quarter,
    "month": lambda dt: dt.dt.month,
    "week": lambda dt: dt.dt.isocalendar().week.astype("Int64"),
    "day": lambda dt: dt.dt.day,
    "dayofweek": lambda dt: dt.dt.dayofweek,
    "dayofyear": lambda dt: dt.dt.dayofyear,
    "hour": lambda dt: dt.dt.hour,
    "is_weekend": lambda dt: dt.dt.dayofweek >= 5,
    "is_month_start": lambda dt: dt.dt.is_month_start,
    "is_month_end": lambda dt: dt.dt.is_month_end,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    """Return a column coerced to float, or raise if it holds no numeric values."""
    if col not in frame.columns:
        raise KeyError(f"No column {col!r}. Known: {list(frame.columns)}")
    series = pd.to_numeric(frame[col], errors="coerce")
    if series.notna().sum() == 0:
        raise ValueError(f"Column {col!r} has no numeric values to compute on.")
    return series.astype(float)


def _clean(series: pd.Series) -> pd.Series:
    """Replace ±inf with NaN so undefined results serialize as missing, not inf."""
    return series.replace([np.inf, -np.inf], np.nan)


def _write(store: Any, name: str, series: pd.Series) -> None:
    """Write a computed Series back as a model-eligible feature column."""
    values = series.tolist()
    if len(values) != len(store.get_frame()):
        raise ValueError("Computed feature length does not match the table row count.")
    store.write_back_column(name, values, feature=True)


# --------------------------------------------------------------------------- #
# 1. arithmetic between two columns
# --------------------------------------------------------------------------- #

def combine_columns(
    store: Any, col_a: str, col_b: str, op: str, name: str | None = None
) -> Result:
    """Create a feature from a binary arithmetic op on two numeric columns.

    ``op`` is one of add, subtract, multiply, divide, ratio. Division-by-zero
    yields NaN. This is the primitive behind most domain features (e.g. density =
    mass / volume): the agent picks the columns and op; the arithmetic is generic.
    """
    if op not in _BINARY_OPS:
        raise ValueError(f"Unknown op {op!r}. Allowed: {sorted(_BINARY_OPS)}.")
    frame = store.get_frame()
    a, b = _numeric(frame, col_a), _numeric(frame, col_b)
    result = _clean(_BINARY_OPS[op](a, b))
    col = name or f"{col_a}_{op}_{col_b}"
    _write(store, col, result)
    return Result(
        method="combine_columns",
        summary=f"Created {col!r} = {col_a} {op} {col_b} ({int(result.notna().sum())} non-null)",
        values={"column": col, "n_non_null": int(result.notna().sum())},
        metadata={"op": op, "inputs": [col_a, col_b], "feature": True},
        trust=_deterministic_trust(
            "Inputs were coerced to numeric and undefined results (e.g. divide-by-zero) "
            "became NaN — some rows may be missing in the new column."
        ),
    )


# --------------------------------------------------------------------------- #
# 2. unary math transform of one column
# --------------------------------------------------------------------------- #

def transform_column(
    store: Any, column: str, func: str, name: str | None = None
) -> Result:
    """Create a feature by applying a math transform to one numeric column.

    ``func`` is one of log, log1p, sqrt, square, reciprocal, abs, zscore. Values
    outside a transform's domain (e.g. log of a non-positive) become NaN.
    """
    if func not in _UNARY_FUNCS:
        raise ValueError(f"Unknown func {func!r}. Allowed: {sorted(_UNARY_FUNCS)}.")
    frame = store.get_frame()
    series = _numeric(frame, column)
    result = _clean(_UNARY_FUNCS[func](series))
    col = name or f"{func}_{column}"
    _write(store, col, result)
    return Result(
        method="transform_column",
        summary=f"Created {col!r} = {func}({column}) ({int(result.notna().sum())} non-null)",
        values={"column": col, "n_non_null": int(result.notna().sum())},
        metadata={"func": func, "input": column, "feature": True},
        trust=_deterministic_trust(
            "Input was coerced to numeric and out-of-domain values (e.g. log of a "
            "non-positive) became NaN — some rows may be missing in the new column."
        ),
    )


# --------------------------------------------------------------------------- #
# 3. binning / discretisation
# --------------------------------------------------------------------------- #

def bin_column(
    store: Any,
    column: str,
    n_bins: int = 4,
    strategy: str = "quantile",
    name: str | None = None,
) -> Result:
    """Discretise a numeric column into ``n_bins`` ordinal bins.

    ``strategy`` = "quantile" (equal-frequency, via qcut) or "uniform"
    (equal-width, via cut). The new column holds integer bin indices (0-based),
    NaN where the source is missing. Useful for turning a skewed continuous field
    into a categorical feature or for coarse grouping.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError(f"Unknown strategy {strategy!r}. Use 'quantile' or 'uniform'.")
    frame = store.get_frame()
    series = _numeric(frame, column)
    try:
        if strategy == "quantile":
            binned = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
        else:
            binned = pd.cut(series, bins=n_bins, labels=False)
    except ValueError as exc:  # e.g. too few distinct values for the bin count
        raise ValueError(f"Could not bin {column!r} into {n_bins} bins: {exc}") from exc
    result = pd.Series(binned, index=series.index).astype("Int64")
    col = name or f"{column}_bin"
    _write(store, col, result)
    n_bins_made = int(result.dropna().nunique())
    return Result(
        method="bin_column",
        summary=f"Created {col!r}: {column} → {n_bins_made} {strategy} bins",
        values={"column": col, "n_bins": n_bins_made},
        metadata={"strategy": strategy, "requested_bins": n_bins, "input": column, "feature": True},
        trust=_deterministic_trust(
            "Input was coerced to numeric (non-numeric rows become NaN) and quantile "
            "binning may collapse fewer bins than requested when values repeat."
        ),
    )


# --------------------------------------------------------------------------- #
# 4. datetime expansion
# --------------------------------------------------------------------------- #

def expand_datetime(
    store: Any, column: str, parts: list[str] | None = None
) -> Result:
    """Expand a datetime column into calendar-component feature columns.

    ``parts`` selects which components to extract (default: year, month,
    dayofweek, is_weekend); any subset of: year, quarter, month, week, day,
    dayofweek, dayofyear, hour, is_weekend, is_month_start, is_month_end. Each
    becomes ``<column>_<part>``. The source is parsed with pandas' datetime
    inference; unparseable rows become NaT and yield NaN components.
    """
    parts = parts or ["year", "month", "dayofweek", "is_weekend"]
    unknown = [p for p in parts if p not in _DATETIME_PARTS]
    if unknown:
        raise ValueError(f"Unknown datetime parts {unknown}. Allowed: {sorted(_DATETIME_PARTS)}.")
    frame = store.get_frame()
    if column not in frame.columns:
        raise KeyError(f"No column {column!r}. Known: {list(frame.columns)}")
    dt = pd.to_datetime(frame[column], errors="coerce")
    if dt.notna().sum() == 0:
        raise ValueError(f"Column {column!r} could not be parsed as datetimes.")

    created = []
    for part in parts:
        col = f"{column}_{part}"
        _write(store, col, _DATETIME_PARTS[part](dt))
        created.append(col)
    return Result(
        method="expand_datetime",
        summary=f"Expanded {column} into {len(created)} features: {', '.join(created)}",
        values={"columns": created},
        metadata={"input": column, "parts": parts, "feature": True},
        trust=_deterministic_trust(
            "The source was parsed as datetimes; unparseable rows became NaT and yield "
            "NaN components — some rows may be missing in the new columns."
        ),
    )


# --------------------------------------------------------------------------- #
# 5. group aggregate (broadcast a per-group statistic back to each row)
# --------------------------------------------------------------------------- #

def group_aggregate(
    store: Any,
    group_by: str,
    value: str,
    agg: str = "mean",
    name: str | None = None,
    add_deviation: bool = False,
) -> Result:
    """Aggregate ``value`` within each ``group_by`` category, broadcast to rows.

    Every row gets its group's statistic (e.g. each order gets its customer's mean
    spend) — a strong, generic relational feature. ``agg`` is one of mean, sum,
    min, max, std, median, count. With ``add_deviation=True`` a second column
    ``<value>_dev_from_<group_by>`` holds value − group-mean, capturing how far
    each row sits from its peers.
    """
    if agg not in _AGGS:
        raise ValueError(f"Unknown agg {agg!r}. Allowed: {sorted(_AGGS)}.")
    frame = store.get_frame()
    for c in (group_by, value):
        if c not in frame.columns:
            raise KeyError(f"No column {c!r}. Known: {list(frame.columns)}")
    values = frame[value] if agg == "count" else _numeric(frame, value)
    grouped = values.groupby(frame[group_by])
    broadcast = grouped.transform(agg)

    col = name or f"{value}_{agg}_by_{group_by}"
    _write(store, col, _clean(pd.Series(broadcast, index=frame.index).astype(float)))
    created = [col]

    if add_deviation:
        dev_col = f"{value}_dev_from_{group_by}"
        group_mean = _numeric(frame, value).groupby(frame[group_by]).transform("mean")
        _write(store, dev_col, _clean(_numeric(frame, value) - group_mean))
        created.append(dev_col)

    return Result(
        method="group_aggregate",
        summary=f"Created {', '.join(created)} ({agg} of {value} within {group_by})",
        values={"columns": created},
        metadata={"group_by": group_by, "value": value, "agg": agg, "feature": True},
        trust=_deterministic_trust(
            "The value column was coerced to numeric (non-numeric rows become NaN) "
            "before aggregating within each group."
        ),
    )


# --------------------------------------------------------------------------- #
# 6. row aggregate across several columns
# --------------------------------------------------------------------------- #

def row_aggregate(
    store: Any, columns: list[str], agg: str = "sum", name: str | None = None
) -> Result:
    """Aggregate several numeric columns across each row into one feature.

    ``agg`` is one of mean, sum, min, max, std, median, count (count = number of
    non-null among the inputs). The generic form of totals like the NOMAD "total
    atom count" from per-element count columns.
    """
    if agg not in _AGGS:
        raise ValueError(f"Unknown agg {agg!r}. Allowed: {sorted(_AGGS)}.")
    if len(columns) < 2:
        raise ValueError("row_aggregate needs at least 2 columns.")
    frame = store.get_frame()
    sub = pd.concat([_numeric(frame, c) for c in columns], axis=1)
    result = sub.notna().sum(axis=1).astype(float) if agg == "count" else sub.agg(agg, axis=1)
    col = name or f"{agg}_of_{len(columns)}_cols"
    _write(store, col, _clean(pd.Series(result, index=frame.index)))
    return Result(
        method="row_aggregate",
        summary=f"Created {col!r} = row-wise {agg} of {len(columns)} columns",
        values={"column": col, "inputs": list(columns)},
        metadata={"agg": agg, "inputs": list(columns), "feature": True},
        trust=_deterministic_trust(
            "Inputs were coerced to numeric (non-numeric values become NaN) before "
            "the row-wise aggregate."
        ),
    )


# --------------------------------------------------------------------------- #
# 7. count → fraction normalisation
# --------------------------------------------------------------------------- #

def normalize_fractions(
    store: Any, columns: list[str], suffix: str = "_frac"
) -> Result:
    """Turn a set of count/amount columns into per-row fractions of their total.

    Each ``<col>`` becomes ``<col><suffix>`` = col / (row sum across the set), so
    the new columns sum to 1 per row (NaN where the row total is 0). The generic
    form of the NOMAD "fraction of each metal" composition features.
    """
    if len(columns) < 2:
        raise ValueError("normalize_fractions needs at least 2 columns.")
    frame = store.get_frame()
    sub = pd.concat([_numeric(frame, c) for c in columns], axis=1)
    total = sub.sum(axis=1)
    created = []
    for c in columns:
        frac = _clean(_numeric(frame, c) / total)
        col = f"{c}{suffix}"
        _write(store, col, frac)
        created.append(col)
    return Result(
        method="normalize_fractions",
        summary=f"Created {len(created)} fraction features from {len(columns)} count columns",
        values={"columns": created},
        metadata={"inputs": list(columns), "feature": True},
        trust=_deterministic_trust(
            "Inputs were coerced to numeric (non-numeric values become NaN) and rows "
            "whose total is 0 yield NaN fractions."
        ),
    )


# --------------------------------------------------------------------------- #
# 8. custom SQL expression — the escape hatch for agent-authored features
# --------------------------------------------------------------------------- #
#
# When the fixed primitives above can't express a feature, the agent supplies a
# single SQL SCALAR EXPRESSION over the table's columns and it is materialized as
# a new column *inside DuckDB* — the computation runs where the data lives, so it
# scales to arbitrarily large tables without streaming rows to the app process.
#
# "Feature generation and nothing else" is enforced structurally: the expression
# is spliced only into a SELECT-list position, statement-chaining (``;``) and
# subqueries / file-and-catalog functions are rejected, and the expression must
# bind against the real table columns. It can therefore only produce one column's
# values, which are then registered as a model-eligible feature.


def compute_feature(store: Any, name: str, expression: str) -> Result:
    """Create one feature column from an agent-authored SQL scalar expression.

    ``expression`` is a DuckDB scalar expression over the table's columns,
    evaluated per row *inside the database* (no data is pulled into the app), and
    stored as a new model-eligible column ``name``. Examples:
      - ``mass / NULLIF(volume, 0)``
      - ``CASE WHEN age >= 18 THEN 'adult' ELSE 'minor' END``
      - ``avg(spend) OVER (PARTITION BY customer_id)``
      - ``regexp_extract(email, '@(.*)$', 1)``

    Only a scalar expression is accepted: statement-chaining, subqueries, and
    file/catalog functions (read_csv, attach, install, ...) are rejected, and the
    expression must reference existing columns. Use this only when the fixed
    primitives (combine_columns, transform_column, group_aggregate, ...) can't
    express the feature.
    """
    n_non_null = store.add_computed_column(name, expression, feature=True)
    return Result(
        method="compute_feature",
        summary=f"Created {name!r} from a SQL expression ({n_non_null} non-null)",
        values={"column": name, "n_non_null": int(n_non_null)},
        metadata={"expression": expression, "feature": True},
        trust=_deterministic_trust(
            "The SQL expression is evaluated deterministically in-database; rows where "
            "it evaluates to NULL are missing in the new column."
        ),
    )
