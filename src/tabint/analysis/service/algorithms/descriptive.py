"""Descriptive / exploratory analytics family.

profile, detect_outliers, association_matrix. Foundation for everything and the
context a future agent reads to understand a table. Libraries: pandas, numpy,
scipy (skew/kurtosis), with association_matrix delegating to analyze_association.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ....shared import honesty
from ....shared.results import Result
from ..validation.dtypes import classify_column
from .association import analyze_association

# Column types that participate in the pairwise association matrix.
_ASSOCIABLE = {"continuous", "categorical_nominal", "categorical_ordinal"}


def profile(store: Any) -> Result:
    """Profile every column: type, missingness, cardinality, distribution, range.

    Args:
        store: The Store instance holding the table.

    Returns:
        Result whose ``values`` maps each column name to its per-column stats.
    """
    frame = store.get_frame()
    n_rows = int(frame.shape[0])
    columns: dict[str, Any] = {}

    for name in frame.columns:
        col = frame[name]
        kind = classify_column(name, store)
        n_missing = int(col.isna().sum())
        stats_dict: dict[str, Any] = {
            "type": kind,
            "n_missing": n_missing,
            "missing_rate": (n_missing / n_rows) if n_rows else 0.0,
            "n_unique": int(col.nunique(dropna=True)),
        }

        if kind == "continuous":
            numeric = pd.to_numeric(col, errors="coerce").dropna()
            if not numeric.empty:
                stats_dict.update(
                    min=float(numeric.min()),
                    max=float(numeric.max()),
                    mean=float(numeric.mean()),
                    median=float(numeric.median()),
                    std=float(numeric.std()),
                    skew=float(stats.skew(numeric)) if numeric.size > 2 else 0.0,
                )
        else:
            top = col.dropna().value_counts().head(5)
            stats_dict["top_values"] = {str(k): int(v) for k, v in top.items()}

        columns[name] = stats_dict

    # Honesty seam: a profile is a faithful summary of the data as given, so it
    # earns high trust — the only caveat is what a summary can and can't tell you.
    caveats = [
        "This is a straight summary of the data as supplied — it describes what's "
        "there, not whether the values are correct or how they came to be.",
    ]
    high_missing = sorted(
        name for name, s in columns.items() if s.get("missing_rate", 0.0) >= 0.2
    )
    if high_missing:
        shown = ", ".join(high_missing[:5]) + ("…" if len(high_missing) > 5 else "")
        caveats.append(
            f"High missingness (20%+ blank) in: {shown} — stats for those columns "
            "cover only the rows that had a value."
        )
    trust = honesty.Trust(
        level=honesty.TrustLevel.HIGH,
        caveats=caveats,
        basis=[f"n={n_rows}"],
    )

    return Result(
        method="profile",
        summary=f"Profiled {len(columns)} columns over {n_rows} rows",
        values=columns,
        metadata={"n_rows": n_rows, "n_columns": len(columns)},
        trust=trust,
    )


def detect_outliers(store: Any, column: str) -> Result:
    """Flag outliers in a numeric column via IQR and z-score, write the flags back.

    A row is flagged if either method flags it. Two columns are written back so
    follow-ups stay ordinary queries:

    - ``<column>_is_outlier`` (boolean): the union of both methods.
    - ``<column>_outlier_method`` (text): which method(s) flagged the row —
      ``"iqr"``, ``"zscore"``, or ``"both"``; ``None`` for non-outliers. This lets
      an agent ask "show only the z-score-flagged rows" without re-running anything.

    Args:
        store: The Store instance holding the table.
        column: Name of the numeric column to analyze.

    Returns:
        Result with per-method counts, the thresholds used, and the flag column names.
    """
    kind = classify_column(column, store)
    if kind != "continuous":
        raise ValueError(f"detect_outliers needs a continuous column; {column!r} is {kind!r}.")

    frame = store.get_frame()
    values = pd.to_numeric(frame[column], errors="coerce")

    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_flag = (values < low) | (values > high)

    mean, std = values.mean(), values.std()
    z = (values - mean) / std if std and not np.isnan(std) else values * 0
    z_flag = z.abs() > 3.0

    iqr_flag = iqr_flag.fillna(False)
    z_flag = z_flag.fillna(False)
    combined = iqr_flag | z_flag

    # Per-row attribution: which method(s) caught each outlier (None if neither),
    # so "which rows did only z-score flag?" is a plain query on the written column.
    both = iqr_flag & z_flag
    method = np.where(
        both, "both", np.where(iqr_flag, "iqr", np.where(z_flag, "zscore", None))
    )

    flag_col = f"{column}_is_outlier"
    method_col = f"{column}_outlier_method"
    store.write_back_column(flag_col, combined.tolist())
    store.write_back_column(method_col, method.tolist())

    n_both = int(both.sum())

    # Honesty seam: trust scales with how many real values we had to judge against.
    n_usable = int(values.notna().sum())
    trust = honesty.from_sample_size(n_usable, low=30, moderate=100, label="values")
    trust = honesty.with_caveats(
        trust,
        "This flags statistical outliers, not necessarily errors — some are "
        "legitimate extremes worth investigating rather than removing.",
    )

    return Result(
        method="outlier_iqr_zscore",
        summary=(
            f"{int(combined.sum())} outliers in {column} "
            f"(IQR: {int(iqr_flag.sum())}, z-score: {int(z_flag.sum())}, both: {n_both})"
        ),
        values={
            "n_outliers": int(combined.sum()),
            "n_outliers_iqr": int(iqr_flag.sum()),
            "n_outliers_zscore": int(z_flag.sum()),
            "n_outliers_both": n_both,
        },
        metadata={
            "iqr_bounds": [float(low), float(high)],
            "zscore_threshold": 3.0,
            "flag_column": flag_col,
            "method_column": method_col,
        },
        trust=trust,
    )


def association_matrix(store: Any) -> Result:
    """Compute pairwise association strength across all associable column pairs.

    Each pair is routed through analyze_association, so every cell uses the
    dtype-appropriate test. The matrix stores the effect size (a 0..1-ish
    strength) for each pair; the per-pair method used is recorded in metadata.

    Args:
        store: The Store instance holding the table.

    Returns:
        Result with a symmetric strength matrix and the measure/method per pair.
    """
    # Skip derived annotations (outlier flags, cluster labels, predictions): they
    # are computed columns, not real variables, so their association with the
    # source data is an artifact rather than a finding. See write_back_column.
    get_derived = getattr(store, "derived_columns", None)
    derived = get_derived() if get_derived else set()
    cols = [
        c
        for c in store._table.schema()
        if c not in derived and classify_column(c, store) in _ASSOCIABLE
    ]
    # Uncomputed/skipped cells stay None (not np.nan) so Result.values serializes
    # to valid, lossless JSON — a bare NaN token is invalid JSON and model_dump_json
    # would otherwise coerce it to null, erasing the not-computed distinction.
    matrix = {a: {b: (1.0 if a == b else None) for b in cols} for a in cols}
    methods: dict[str, str] = {}

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                res = analyze_association(store, a, b)
                effect = res.values.get("effect_size")
                strength = None if effect is None else float(effect)
                matrix[a][b] = matrix[b][a] = strength
                methods[f"{a}__{b}"] = res.method
            except Exception as exc:  # a single bad pair shouldn't sink the matrix
                methods[f"{a}__{b}"] = f"skipped: {exc}"

    # Honesty seam: trust from the table size (fewer rows → noisier cells), with
    # the association-is-not-causation warning made mandatory for the whole matrix.
    n_rows = int(store.get_frame().shape[0])
    trust = honesty.from_sample_size(n_rows, low=30, moderate=100, label="rows")
    trust = honesty.with_caveats(
        trust,
        "Association is not causation — strong cells show variables that move "
        "together, not one causing another.",
        "Each cell is an effect size on a 0-1 scale; scan for the strong ones, but "
        "confirm any that matter with a focused test on that pair.",
    )

    return Result(
        method="association_matrix",
        summary=f"Pairwise association across {len(cols)} columns",
        values={"columns": cols, "matrix": matrix},
        metadata={"methods": methods},
        trust=trust,
    )
