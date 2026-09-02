"""Column type classification — the single source of truth for dtype routing.

Every analytics function that needs to know "is this column categorical or
continuous?" must call classify_column here. No local re-implementation of
this logic is permitted. This prevents the most common source of inconsistency
in tabular analysis pipelines.

Classification is **cache-and-compute**: the resolved type for every column is
persisted in a sidecar DuckDB table (``_ti_column_types``, keyed by table +
column name) so it survives reopens for free. ``classify_column`` reads that
cache first; on a miss it runs the auto-detector (ibis expressions over DuckDB,
no data materialised into Python) and writes the result back. An analyst/agent
refines a type via ``set_column_type``, which writes a specific value into the
same store. The cache is authoritative: once set, the value is returned as-is.

The auto-detector only ever returns ``continuous`` / ``datetime`` /
``identifier`` / ``categorical``. ``categorical`` is the *coarse, undecided*
label — modeling algorithms refuse to run on it (they require nominal or
ordinal), which forces a refinement via set_column_type before modeling.
``categorical_nominal`` and ``categorical_ordinal`` are the two refinements,
and are the only values set_column_type accepts.
"""
from __future__ import annotations

import ibis.expr.datatypes as dt

# Vocabulary of column types. "categorical" is the auto-detected coarse label
# (category-like, not yet refined); "categorical_nominal" / "categorical_ordinal"
# are the refinements an analyst/agent assigns. The auto-detector never returns
# the latter two.
COLUMN_TYPES = frozenset({
    "continuous",
    "categorical",
    "categorical_nominal",
    "categorical_ordinal",
    "datetime",
    "identifier",
})

# The only values set_column_type (the agent-facing tool) may assign. Auto-
# detected types (continuous/datetime/identifier) are not settable: an agent
# disagreeing with those is expected to transform the column instead.
SETTABLE_TYPES = frozenset({"categorical_nominal", "categorical_ordinal"})

# A string column whose distinct-value ratio exceeds this threshold is treated
# as an identifier (e.g. UUIDs, email addresses, primary keys).
_IDENTIFIER_DISTINCT_RATIO = 0.9

# A numeric column with at most this many distinct values is treated as
# categorical rather than continuous.
_NUMERIC_CATEGORICAL_MAX_DISTINCT = 15

# A real-valued (float/decimal) column whose distinct-value ratio exceeds this is
# treated as continuous even below the absolute cutoff — otherwise a genuinely
# continuous float on a small table gets mislabeled categorical.
_CONTINUOUS_DISTINCT_RATIO = 0.5


def classify_column(col_name: str, store: object, override: bool = False) -> str:
    """Classify a single column into a canonical type (cache-and-compute).

    Reads the persisted type from the sidecar store first; on a miss (or when
    ``override`` is True) runs the auto-detector and stores the result back.
    The auto-detector runs entirely in DuckDB via ibis expressions — no column
    data is ever pulled into Python memory.

    Args:
        col_name: Name of the column to classify.
        store:    A Store instance (provides ``store._table`` ibis TableExpr and
                  the ``get_column_type`` / ``set_column_type`` accessors).
        override: If True, ignore any cached value, recompute from the data, and
                  overwrite the cache. Default False (use the cache). Note this
                  can clobber a value an agent set via set_column_type — that is
                  the point of forcing a recompute.

    Returns:
        One of: "continuous", "categorical", "categorical_nominal",
        "categorical_ordinal", "datetime", "identifier". The auto-detector only
        ever returns the first two/three plus datetime/identifier; nominal and
        ordinal come only from set_column_type.
    """
    if not override:
        get_cached = getattr(store, "get_cached_type", None)
        if get_cached:
            cached = get_cached(col_name)
            if cached:
                return cached

    computed = _detect(col_name, store)

    set_cached = getattr(store, "set_cached_type", None)
    if set_cached:
        set_cached(col_name, computed)
    return computed


def _detect(col_name: str, store: object) -> str:
    """Auto-detect a column's coarse type from its data (no cache, no sidecar)."""
    table = store._table
    dtype = table.schema()[col_name]

    # Temporal types — classify immediately from schema, no query needed.
    if isinstance(dtype, (dt.Timestamp, dt.Date, dt.Time)):
        return "datetime"

    # Boolean — categorical (undecided nominal/ordinal; refine via set_column_type).
    if isinstance(dtype, dt.Boolean):
        return "categorical"

    col = table[col_name]

    # Numeric types.
    if isinstance(dtype, (dt.Integer, dt.Floating, dt.Decimal)):
        n_distinct = col.nunique().execute()
        if n_distinct > _NUMERIC_CATEGORICAL_MAX_DISTINCT:
            return "continuous"
        # A real-valued column with mostly-distinct values is continuous even on a
        # small table, where the absolute cutoff alone would mislabel it.
        if isinstance(dtype, (dt.Floating, dt.Decimal)):
            n_rows = table.count().execute()
            if n_rows > 0 and (n_distinct / n_rows) > _CONTINUOUS_DISTINCT_RATIO:
                return "continuous"
        return "categorical"

    # String types.
    if isinstance(dtype, dt.String):
        n_rows = table.count().execute()
        n_distinct = col.nunique().execute()
        if n_rows > 0 and (n_distinct / n_rows) > _IDENTIFIER_DISTINCT_RATIO:
            return "identifier"
        return "categorical"

    # Fallback for any other type (arrays, structs, etc.).
    return "categorical"


def classify_table(store: object, override: bool = False) -> dict[str, str]:
    """Classify every column in the store's table.

    Args:
        store: A Store instance.
        override: Forwarded to classify_column (force recompute of every column).

    Returns:
        Mapping of column name → type string (same vocabulary as classify_column).
    """
    return {
        col: classify_column(col, store, override=override)
        for col in store._table.schema()
    }
