"""Shared feature-preparation helpers for the model-based families.

Clustering, dimensionality reduction, and supervised learning all need to turn
the stored table into a numeric, model-ready matrix. Doing that consistently —
same column selection, same encoding, same imputation — is what keeps results
comparable across families, so it lives here rather than being re-derived in each
module.

Column *classification* is never re-decided here; it is delegated to
validation.dtypes (the single source of truth). This module only decides how a
classified column is fed to scikit-learn.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .validation.dtypes import classify_column

# Column types (from validation.dtypes) that carry modelling signal and how they
# should be treated. Identifiers and datetimes are excluded from feature matrices.
# Bare "categorical" (the auto-detected, unrefined label) is deliberately NOT a
# member of either set: feature_columns raises on it instead, forcing a refinement
# to nominal/ordinal via set_column_type before any model can use the column.
_NUMERIC_TYPES = {"continuous"}
_CATEGORICAL_TYPES = {"categorical_nominal", "categorical_ordinal"}
_ORDINAL_TYPES = {"categorical_ordinal"}
_UNCLASSIFIED_TYPE = "categorical"


def get_frame(store: Any) -> pd.DataFrame:
    """Return the full table in stable (_ti_row) order as a pandas DataFrame."""
    return store.get_frame()


def feature_columns(
    store: Any,
    exclude: tuple[str, ...] = (),
) -> tuple[list[str], list[str], list[str]]:
    """Split the table's columns into (numeric, nominal, ordinal) feature lists.

    Identifier and datetime columns are dropped — they are not features. Any
    names in ``exclude`` (e.g. the supervised target, or a cluster-label column)
    are dropped too. Ordinal columns (categorical_ordinal) are returned in their
    own list so build_preprocessor can integer-encode them rather than one-hot.

    Raises ValueError if any remaining column is still at the unrefined
    ``categorical`` label — modeling on an undecided column is not allowed; the
    caller (an agent) must refine it via set_column_type first. The error names
    the offending columns so the agent knows exactly what to classify.

    Args:
        store: The Store instance.
        exclude: Column names to omit from all three lists.

    Returns:
        (numeric_columns, nominal_columns, ordinal_columns).
    """
    # Derived annotations (outlier flags, cluster labels, predictions) are never
    # features — drop them alongside the caller's explicit excludes so they can't
    # leak into a model. reduce_dimensions marks its components feature=True, so
    # those stay eligible.
    get_derived = getattr(store, "derived_columns", None)
    excluded = set(exclude) | (get_derived() if get_derived else set())

    numeric: list[str] = []
    nominal: list[str] = []
    ordinal: list[str] = []
    unclassified: list[str] = []
    for name in store._table.schema():
        if name in excluded:
            continue
        kind = classify_column(name, store)
        if kind in _NUMERIC_TYPES:
            numeric.append(name)
        elif kind in _ORDINAL_TYPES:
            ordinal.append(name)
        elif kind in _CATEGORICAL_TYPES:
            nominal.append(name)
        elif kind == _UNCLASSIFIED_TYPE:
            unclassified.append(name)
        # identifier / datetime → intentionally skipped
    if unclassified:
        raise ValueError(
            "These categorical columns are unclassified; call "
            "list_categorical_columns then set_column_type (to categorical_nominal "
            f"or categorical_ordinal) before modelling: {unclassified}"
        )
    return numeric, nominal, ordinal


def build_preprocessor(
    numeric: list[str],
    nominal: list[str],
    ordinal: list[str] | None = None,
    *,
    scale: bool,
) -> ColumnTransformer:
    """Build a ColumnTransformer: numeric + ordinal-encoded + one-hot.

    Three branches: numeric (impute, optionally scale), ordinal (impute, integer
    -encode via OrdinalEncoder, optionally scale — the integer codes need scaling
    for distance-based methods like k-means/PCA), and nominal (impute, one-hot).

    Args:
        numeric: Numeric (continuous) feature column names.
        nominal: Categorical_nominal feature column names (one-hot encoded).
        ordinal: Categorical_ordinal feature column names (integer-encoded).
        scale: Whether to standard-scale numeric and ordinal columns. Distance-
            based methods (k-means, PCA) need this; tree ensembles do not.

    Returns:
        An unfitted ColumnTransformer.
    """
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Dense output: HistGradientBoosting and the manifold methods reject sparse.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal = ordinal or []
    ordinal_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Integer-encode the ordered levels; unseen values at predict time map to
        # -1 so a served row with a new level fails safe rather than crashing.
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
    if scale:
        # Raw integer codes have arbitrary magnitude — scale them so a distance-
        # based model treats the ordinal axis comparably to the numeric ones.
        ordinal_steps.append(("scale", StandardScaler()))
    ordinal_pipe = Pipeline(ordinal_steps)

    transformers = []
    if numeric:
        transformers.append(("numeric", numeric_pipe, numeric))
    if ordinal:
        transformers.append(("ordinal", ordinal_pipe, ordinal))
    if nominal:
        transformers.append(("nominal", nominal_pipe, nominal))
    return ColumnTransformer(transformers, remainder="drop")


def numeric_matrix(
    store: Any,
    exclude: tuple[str, ...] = (),
    *,
    scale: bool = True,
) -> tuple[Any, pd.DataFrame, list[str]]:
    """Materialize a fully numeric feature matrix for distance-based methods.

    Used by clustering and dimensionality reduction. Numeric columns are scaled
    (by default); ordinal columns are integer-encoded and scaled; nominal columns
    are one-hot encoded.

    Args:
        store: The Store instance.
        exclude: Column names to omit (e.g. an existing cluster-label column).
        scale: Whether to standard-scale numeric and ordinal columns.

    Returns:
        (X, frame, feature_names) where X is the transformed 2-D array in stable
        row order, frame is the source DataFrame, and feature_names are the
        original feature column names (pre-encoding).
    """
    frame = get_frame(store)
    numeric, nominal, ordinal = feature_columns(store, exclude=exclude)
    if not numeric and not nominal and not ordinal:
        raise ValueError("No usable feature columns (all identifier/datetime).")
    pre = build_preprocessor(numeric, nominal, ordinal, scale=scale)
    X = pre.fit_transform(frame)
    # Densify sparse one-hot output so downstream estimators that dislike sparse
    # input (e.g. some sklearn manifold methods) work uniformly.
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X, frame, numeric + nominal + ordinal
