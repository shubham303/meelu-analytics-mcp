"""Model interpretation family.

feature_importance uses permutation importance (model-agnostic, computed on the
held-out test split) over the original feature columns. explain_prediction uses
SHAP for a single row. Libraries: scikit-learn (permutation), shap.

Foundation-model backends (TabICL) re-read the whole training context on every
predict call, and score only ~40–150 rows/s on a laptop CPU. Both explainers
therefore (1) run on a 2-member refit of the same model (``_explainer_pipeline``)
rather than the full ensemble, (2) score every perturbed copy in one batched
predict call, and (3) cap the TOTAL rows scored per call (``_FOUNDATION_SCORED_ROWS``)
so the call stays within about a minute — declining, with the reason, when the
table is too wide to explain inside that budget. Budgets go in the metadata.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from ....shared import honesty
from ....shared.identity import _lazy_import
from ....shared.results import Result

# shap pulls in numba/llvmlite; keep it lazy so importing this module is cheap
# and the rest of the library works even if shap is unavailable.
shap = _lazy_import("shap")

_RANDOM_STATE = 0
_N_REPEATS = 10
# Foundation models (see the module docstring). Total rows one explainer call may
# score across all perturbed copies — ~30–40 s at the 2-member refit's CPU speed.
_FOUNDATION_SCORED_ROWS = 4_000
_FOUNDATION_N_REPEATS = 3
_FOUNDATION_MIN_IMPORTANCE_ROWS = 30
_FOUNDATION_SHAP_BACKGROUND = 20
_FOUNDATION_MIN_SHAP_BACKGROUND = 5


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
    foundation_model = getattr(model, "is_foundation", False)
    extra_meta: dict[str, Any] = {}
    if foundation_model:
        n_repeats = _FOUNDATION_N_REPEATS
        copies = 1 + len(model._feature_names) * n_repeats
        n_scored = min(len(X_test), _FOUNDATION_SCORED_ROWS // copies)
        if n_scored < _FOUNDATION_MIN_IMPORTANCE_ROWS:
            return _declined_explainer(
                "permutation_importance", model,
                f"{len(model._feature_names)} features × {n_repeats} shuffles would need "
                f"{copies * _FOUNDATION_MIN_IMPORTANCE_ROWS:,} foundation-model predictions, "
                f"above the per-call CPU budget of {_FOUNDATION_SCORED_ROWS:,}. Retrain with "
                "backend='gbt' to rank features on a table this wide.")
        if len(X_test) > n_scored:
            keep = np.sort(np.random.RandomState(_RANDOM_STATE).choice(
                len(X_test), n_scored, replace=False))
            X_test, y_test = X_test.iloc[keep], y_test.iloc[keep]
        pipeline, n_est = _explainer_pipeline(model)
        importances = _batched_permutation_importance(
            pipeline, model._feature_names, model._task, X_test, y_test, n_repeats)
        extra_meta = {"explainer_n_estimators": n_est, "rows_scored_total": n_scored * copies}
    else:
        n_repeats = _N_REPEATS
        result = permutation_importance(
            model._pipeline, X_test, y_test,
            n_repeats=n_repeats, random_state=_RANDOM_STATE,
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
    caveats = ["Importance shows what the MODEL relied on, not what CAUSES the outcome — "
               "important features can be proxies or correlated with the real driver."]
    if foundation_model:
        caveats.append(
            f"Foundation model: measured on {n_test} held-out rows with {n_repeats} shuffles "
            "each, using a smaller ensemble of the same model to fit the CPU budget — "
            "small differences between features are noise.")
    trust = honesty.with_caveats(base, *caveats)

    return Result(
        method="permutation_importance",
        summary=f"Top feature: {top}" if top else "No features",
        values={"importances": ranked},
        metadata={
            "target": model._target,
            "task": model._task,
            "backend": getattr(model, "_backend", "gbt"),
            "n_repeats": n_repeats,
            "n_rows_scored": n_test,
            "measure": "mean_score_decrease",
            **extra_meta,
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

    foundation_model = getattr(model, "is_foundation", False)
    pipeline = model._pipeline
    n_est = None
    if foundation_model:
        n_encoded = len(model._pipeline.named_steps["pre"].get_feature_names_out())
        evals = 2 * n_encoded + 1
        if _FOUNDATION_SCORED_ROWS // evals < _FOUNDATION_MIN_SHAP_BACKGROUND:
            return _declined_explainer(
                "shap_permutation", model,
                f"{n_encoded} encoded features need {evals} SHAP evaluations per background "
                f"row; even a {_FOUNDATION_MIN_SHAP_BACKGROUND}-row background exceeds the "
                f"per-call CPU budget of {_FOUNDATION_SCORED_ROWS:,} foundation-model "
                "predictions. Retrain with backend='gbt' for exact explanations on a "
                "table this wide.")
        pipeline, n_est = _explainer_pipeline(model)
    pre = pipeline.named_steps["pre"]
    estimator = pipeline.named_steps["model"]
    background = pre.transform(model._X_test[features])
    x_row = pre.transform(row_df)
    encoded_names = list(pre.get_feature_names_out())

    if foundation_model:
        explanation, budget = _foundation_shap(model, estimator, background, x_row, encoded_names)
        budget = f"{budget}, {n_est}-member ensemble"
        method, basis = "shap_permutation", f"permutation SHAP, {budget}"
    else:
        # TreeExplainer is exact and fast for the gradient-boosted default; fall
        # back to the model-agnostic explainer if a non-tree estimator is swapped in.
        method, basis, budget = "shap", "local SHAP explanation", None
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
    # Multiclass → collapse to the class with the largest total contribution
    # (argmax takes the first on a tie, so the pick is deterministic).
    classes = list(getattr(estimator, "classes_", []))
    explained_class = classes[1] if len(classes) == 2 else None
    if values.ndim > 1:
        cls = int(np.argmax(np.abs(values).sum(axis=0)))
        values = values[:, cls]
        base = base[cls] if np.ndim(base) else base
        explained_class = classes[cls] if cls < len(classes) else cls

    contributions = _aggregate_to_columns(
        pre, values,
        model._numeric_features, model._nominal_features, model._ordinal_features,
    )
    ranked = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

    # Honesty seam — a local explanation is well-grounded (exact SHAP on this row),
    # but it is specific to this row, not a global rule. Moderate by default.
    caveats = ["This explains ONE prediction locally (SHAP) — it's specific to this row, "
               "not a global rule."]
    if budget:
        caveats.append(
            f"Approximate SHAP for a foundation model ({budget}) — contribution sizes are "
            "estimates; their ranking is more reliable than their exact values."
        )
    trust = honesty.with_caveats(
        honesty.Trust(level=honesty.TrustLevel.MODERATE, basis=[basis]),
        *caveats,
    )

    metadata = {"target": model._target, "task": model._task,
                "backend": getattr(model, "_backend", "gbt")}
    if explained_class is not None:
        metadata["explained_class"] = explained_class
    if budget:
        metadata["approximation"] = budget
    return Result(
        method=method,
        summary=f"Top driver: {next(iter(ranked), None)}",
        values={"contributions": ranked, "base_value": float(np.ravel(base)[0])},
        metadata=metadata,
        trust=trust,
    )


def _declined_explainer(method: str, model: Any, reason: str) -> Result:
    return Result(
        method=method,
        summary=f"Declined: {reason}",
        values={},
        metadata={"target": model._target, "task": model._task,
                  "backend": getattr(model, "_backend", "gbt"),
                  "scored_rows_budget": _FOUNDATION_SCORED_ROWS},
        trust=honesty.decline(reason),
    )


def _explainer_pipeline(model: Any) -> tuple[Any, int | None]:
    """The pipeline the foundation explainers score: a 2-member refit of the same
    in-context model on its stored training split (about 4x cheaper per call
    than the full ensemble). Fine-tuned models — whose weights a refit would
    lose — and models saved without their training split use the model as is."""
    from . import foundation

    estimator = model._pipeline.named_steps["model"]
    X_train = getattr(model, "_X_train", None)
    if getattr(model, "_backend", None) != "tabicl" or X_train is None:
        return model._pipeline, getattr(estimator, "n_estimators",
                                        getattr(estimator, "n_estimators_inference", None))
    pipeline = clone(model._pipeline)
    pipeline.set_params(model__n_estimators=foundation.N_ESTIMATORS_EXPLAIN)
    with foundation.quiet():
        pipeline.fit(X_train, model._y_train)
    return pipeline, foundation.N_ESTIMATORS_EXPLAIN


def _batched_permutation_importance(
    pipeline: Any, feature_names: list[str], task: str,
    X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int,
) -> dict[str, float]:
    """Permutation importance with every shuffled copy scored in one predict call.

    Same quantity as sklearn's ``permutation_importance`` with the estimator's
    default score (accuracy / R²): the mean score drop when one original column
    is shuffled. Batching matters because a foundation model's cost is per call,
    not per row.
    """
    from sklearn.metrics import accuracy_score, r2_score

    from . import foundation

    score = accuracy_score if task == "classification" else r2_score
    rng = np.random.RandomState(_RANDOM_STATE)
    X_base = X_test[feature_names].reset_index(drop=True)
    y_true = np.asarray(y_test)
    n = len(X_base)

    copies = [X_base]
    for feat in feature_names:
        for _ in range(n_repeats):
            shuffled = X_base.copy()
            # .iloc keeps the column's dtype (categorical/extension types survive).
            shuffled[feat] = X_base[feat].iloc[rng.permutation(n)].to_numpy()
            shuffled[feat] = shuffled[feat].astype(X_base[feat].dtype)
            copies.append(shuffled)
    # Total rows are capped by the caller, so one predict call scores them all.
    with foundation.quiet():
        preds = np.asarray(pipeline.predict(pd.concat(copies, ignore_index=True)))

    baseline = score(y_true, preds[:n])
    importances: dict[str, float] = {}
    for i, feat in enumerate(feature_names):
        drops = []
        for r in range(n_repeats):
            start = n * (1 + i * n_repeats + r)
            drops.append(baseline - score(y_true, preds[start:start + n]))
        importances[feat] = float(np.mean(drops))
    return importances


def _foundation_shap(model: Any, estimator: Any, background: np.ndarray, x_row: np.ndarray,
                     encoded_names: list[str]):
    """Permutation SHAP over a small fixed background, in a handful of predict calls."""
    from . import foundation

    rng = np.random.RandomState(_RANDOM_STATE)
    evals = 2 * len(encoded_names) + 1
    size = min(_FOUNDATION_SHAP_BACKGROUND, _FOUNDATION_SCORED_ROWS // evals)
    if len(background) > size:
        background = background[rng.choice(len(background), size, replace=False)]
    if model._task == "classification" and len(estimator.classes_) == 2:
        # Binary: explain P(positive class) — the same quantity TreeExplainer
        # explains for trees. Explaining both columns would tie (they are exact
        # negatives) and the "dominant" class would be decided by float noise.
        f = lambda data: estimator.predict_proba(data)[:, 1]
    elif model._task == "classification":
        f = lambda data: estimator.predict_proba(data)
    else:
        f = lambda data: estimator.predict(data)
    # The permutation explainer needs at least 2·features+1 evaluations for one
    # pass; a large batch size turns those into very few model calls.
    max_evals = 2 * len(encoded_names) + 1
    explainer = shap.PermutationExplainer(
        f, shap.maskers.Independent(background, max_samples=len(background)),
        feature_names=encoded_names, seed=_RANDOM_STATE,
    )
    with foundation.quiet():
        explanation = explainer(x_row, max_evals=max_evals, batch_size=max_evals)
    budget = f"{len(background)}-row background, {max_evals} evaluations"
    return explanation, budget


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
