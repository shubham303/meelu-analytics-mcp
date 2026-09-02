"""Key-driver / segment explainer — the "why is this metric high/low" primitive.

``explain_metric`` fits a *shallow* decision tree (sklearn) with the target as the
label and every other usable column as a feature. The tree does the heavy lifting:

* ``feature_importances_`` → a ranked list of the columns that most drive the metric
  (the key-driver answer);
* ``export_text`` → the human-readable segment rules ("monetary > 500 AND
  region = NE → churn rate 0.08"), which is what actually goes in a client finding.

We keep the tree shallow on purpose: the value is an interpretable set of segments,
not a black-box predictor (use train_classifier for that). Categoricals are
ordinal-encoded so each feature stays a single, named column the rules can reference.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from ....shared import honesty
from ....shared.results import Result
from ..validation.dtypes import classify_column
from .. import _prep

_MAX_DEPTH = 3
_MIN_LEAF_FRACTION = 0.05  # a segment must cover >=5% of rows to be worth reporting
_RANDOM_STATE = 0


def explain_metric(store: Any, target: str, max_depth: int = _MAX_DEPTH) -> Result:
    """Explain what drives ``target``: ranked drivers + interpretable segment rules.

    Args:
        store: The Store/Table instance.
        target: The metric/column to explain (numeric → regression tree; otherwise
            a classification tree).
        max_depth: Depth of the explanatory tree (shallower = simpler segments).

    Returns:
        Result with ``drivers`` (feature → importance, sorted), ``rules`` (text),
        and ``explained`` (tree R²/accuracy — how much of the metric the segments
        account for).
    """
    frame = store.get_frame()
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} not in table.")

    numeric, nominal, ordinal = _prep.feature_columns(store, exclude=(target,))
    categorical = nominal + ordinal
    features = numeric + categorical
    if not features:
        raise ValueError("No usable feature columns to explain the metric with.")

    y = frame[target]
    rows = y.notna()
    X, y = frame.loc[rows, features], y[rows]
    if y.empty:
        raise ValueError(f"Target column {target!r} has no non-null values.")

    is_regression = classify_column(target, store) == "continuous"

    # Ordinal-encode categoricals so each stays one named column (readable rules);
    # impute so the tree never chokes on gaps. Numeric passes through untouched.
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat", cat_pipe, categorical))
    pre = ColumnTransformer(transformers, remainder="drop")
    ordered_features = numeric + categorical  # ColumnTransformer output order

    min_leaf = max(1, int(_MIN_LEAF_FRACTION * len(y)))
    tree_cls = DecisionTreeRegressor if is_regression else DecisionTreeClassifier
    tree = tree_cls(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=_RANDOM_STATE)

    Xe = pre.fit_transform(X)
    tree.fit(Xe, y)

    importances = dict(
        sorted(
            ((f, float(i)) for f, i in zip(ordered_features, tree.feature_importances_)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    rules = export_text(tree, feature_names=list(ordered_features))
    explained = float(tree.score(Xe, y))  # R² (regression) or accuracy (classification)

    top = next(iter(importances), None)
    kind = "R²" if is_regression else "accuracy"
    summary = (
        f"Top driver of {target!r}: {top!r}" if top else f"No clear driver of {target!r}"
    ) + f" (tree {kind}={explained:.2f})"

    # Honesty seam: confidence from data volume, plus the mandatory caveat that this
    # is attribution (which columns co-move with the metric), not causal explanation.
    trust = honesty.from_sample_size(int(len(y)), low=30, moderate=200, label="rows")
    trust = honesty.with_caveats(
        trust,
        "This is arithmetic attribution (which components moved the number), NOT a "
        "causal explanation of why they moved.",
    )

    return Result(
        method="decision_tree_key_drivers",
        summary=summary,
        values={"drivers": importances, "rules": rules, "explained": explained},
        trust=trust,
        metadata={
            "target": target,
            "task": "regression" if is_regression else "classification",
            "max_depth": max_depth,
            "n_rows": int(len(y)),
            "features": ordered_features,
        },
    )
