"""Model interpretation family.

feature_importance uses permutation importance (model-agnostic, computed on the
held-out test split) over the original feature columns. explain_prediction uses
SHAP for a single row. Libraries: scikit-learn (permutation), shap.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ....shared import honesty
from ....shared.identity import _lazy_import
from ....shared.results import Result

# shap pulls in numba/llvmlite; keep it lazy so importing this module is cheap
# and the rest of the library works even if shap is unavailable.
shap = _lazy_import("shap")

_RANDOM_STATE = 0


def feature_importance(model: Any) -> Result:
    """Compute permutation feature importance on the model's held-out test split.

    Permutation importance is model-agnostic and measured over the *original*
    feature columns (before one-hot expansion), so each score maps to a column a
    user recognises. Scores are the mean drop in model score when that column is
    shuffled, sorted descending.

    Args:
        model: A TrainedModel from train_classifier / train_regressor.

    Returns:
        Result with per-feature importance, sorted most-important first.
    """
    X_test, y_test = model._X_test, model._y_test
    result = permutation_importance(
        model._pipeline, X_test, y_test,
        n_repeats=10, random_state=_RANDOM_STATE,
    )
    importances = {
        feat: float(mean)
        for feat, mean in zip(model._feature_names, result.importances_mean)
    }
    ranked = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    top = next(iter(ranked), None)

    # Honesty seam — inherit the model's trust (importance is only as reliable as
    # the model) and always attach the correlation-vs-causation caveat.
    n_test = int(len(y_test))
    base = honesty.combine(
        getattr(model, "_trust", None),
        honesty.from_sample_size(n_test, low=30, moderate=100, label="test rows"),
    )
    trust = honesty.with_caveats(
        base,
        "Importance shows what the MODEL relied on, not what CAUSES the outcome — "
        "important features can be proxies or correlated with the real driver.",
    )

    return Result(
        method="permutation_importance",
        summary=f"Top feature: {top}" if top else "No features",
        values={"importances": ranked},
        metadata={
            "target": model._target,
            "task": model._task,
            "n_repeats": 10,
            "measure": "mean_score_decrease",
        },
        trust=trust,
    )


def explain_prediction(model: Any, row: Any) -> Result:
    """Explain a single prediction with SHAP, aggregated to original columns.

    SHAP runs in the model's *transformed* (fully numeric, one-hot expanded)
    feature space — this avoids the mixed string/number masking problems of
    explaining a raw pipeline. Contributions from one-hot columns are then summed
    back to the original column they came from, so each score maps to a column a
    user recognises.

    Args:
        model: A TrainedModel from train_classifier / train_regressor.
        row: A single row as a pandas Series, dict, or 1-row DataFrame.

    Returns:
        Result with per-feature SHAP contributions and the base value.
    """
    features = model._feature_names
    row_df = _row_to_frame(row, features)

    pre = model._pipeline.named_steps["pre"]
    estimator = model._pipeline.named_steps["model"]
    background = pre.transform(model._X_test[features])
    x_row = pre.transform(row_df)
    encoded_names = list(pre.get_feature_names_out())

    # TreeExplainer is exact and fast for the gradient-boosted default; fall back
    # to the model-agnostic explainer if a non-tree estimator is ever swapped in.
    try:
        explainer = shap.TreeExplainer(estimator, background, feature_names=encoded_names)
        explanation = explainer(x_row)
    except Exception:
        if model._task == "classification":
            f = lambda data: estimator.predict_proba(data)
        else:
            f = lambda data: estimator.predict(data)
        explainer = shap.Explainer(f, background, feature_names=encoded_names)
        explanation = explainer(x_row)

    values = np.asarray(explanation.values)[0]
    base = np.asarray(explanation.base_values)[0]
    # Multiclass → collapse to the class with the largest total contribution.
    if values.ndim > 1:
        cls = int(np.argmax(np.abs(values).sum(axis=0)))
        values = values[:, cls]
        base = base[cls] if np.ndim(base) else base

    contributions = _aggregate_to_columns(
        pre, values,
        model._numeric_features, model._nominal_features, model._ordinal_features,
    )
    ranked = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

    # Honesty seam — a local explanation is well-grounded (exact SHAP on this row),
    # but it is specific to this row, not a global rule. Moderate by default.
    trust = honesty.with_caveats(
        honesty.Trust(level=honesty.TrustLevel.MODERATE, basis=["local SHAP explanation"]),
        "This explains ONE prediction locally (SHAP) — it's specific to this row, "
        "not a global rule.",
    )

    return Result(
        method="shap",
        summary=f"Top driver: {next(iter(ranked), None)}",
        values={"contributions": ranked, "base_value": float(np.ravel(base)[0])},
        metadata={"target": model._target, "task": model._task},
        trust=trust,
    )


def _aggregate_to_columns(
    pre: Any,
    values: np.ndarray,
    numeric: list[str],
    nominal: list[str],
    ordinal: list[str],
) -> dict[str, float]:
    """Sum encoded-feature SHAP values back onto their original source columns.

    Uses the *structure* of the fitted ColumnTransformer rather than parsing
    concatenated feature-name strings (which mis-attributes when a column name
    plus a category value collides with another column, e.g. 'a' vs 'a_b').

    The transformed layout, matching build_preprocessor, is three blocks in
    order: numeric (one output per column), ordinal (one output per column,
    integer-encoded), nominal (one-hot, len(categories_[i]) outputs per column).
    A block that was empty at fit time is absent from the transformer, so the
    walk keys off the transformer's named branches rather than assuming all
    three are present.
    """
    totals = {col: 0.0 for col in numeric + nominal + ordinal}
    idx = 0
    branches = pre.named_transformers_
    # Numeric block: one SHAP value per numeric column, in order.
    if numeric and "numeric" in branches:
        for col in numeric:
            totals[col] += float(values[idx])
            idx += 1
    # Ordinal block: one SHAP value per ordinal column (integer-encoded → 1 out).
    if ordinal and "ordinal" in branches:
        for col in ordinal:
            totals[col] += float(values[idx])
            idx += 1
    # Nominal block: one-hot, len(categories_[i]) outputs per nominal column.
    if nominal and "nominal" in branches:
        ohe = branches["nominal"].named_steps["onehot"]
        for i, col in enumerate(nominal):
            for _ in range(len(ohe.categories_[i])):
                totals[col] += float(values[idx])
                idx += 1
    return totals


def _row_to_frame(row: Any, features: list[str]) -> pd.DataFrame:
    if isinstance(row, pd.DataFrame):
        frame = row.copy()
    elif isinstance(row, pd.Series):
        frame = row.to_frame().T
    elif isinstance(row, dict):
        frame = pd.DataFrame([row])
    else:
        raise TypeError("row must be a DataFrame, Series, or dict.")
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return frame[features]
