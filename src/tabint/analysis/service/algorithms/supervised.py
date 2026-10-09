"""Supervised learning family — selectable backends, one deterministic default.

train_classifier / train_regressor build a scikit-learn Pipeline (shared
preprocessing + an estimator), fit it on a proper train/test split, and return a
TrainedModel that bundles the fitted preprocessing so new rows are transformed
identically at predict time (no train/serve skew).

Backends (``backend=``):

* ``"auto"`` (the MCP tools' default) — picks one of the two below by a fixed
  rule on the installed extras and the table's shape (see
  ``foundation.select_backend``). The choice and the reason are recorded in
  ``TrainedModel._selection``.
* ``"gbt"`` (the Python API's default, unchanged) — sklearn-native HistGradientBoosting. No fragile system deps, strong
  on tabular data, works on CPU at any size.
* ``"tabicl"`` — TabICL v2, a tabular *foundation model*. One forward pass through
  a pre-trained transformer (in-context learning, no per-task gradient training);
  often beats tuned trees out of the box on small/medium tables. BSD-3 code and
  weights. Installed on demand (``foundation_install``), never by default.

``finetune`` additionally adapts the foundation model's weights to one table
(``"tabicl_finetuned"``) under a hard CPU time budget.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ....shared import honesty
from ....shared.results import Result
from ..validation.dtypes import classify_column
from .. import _prep
from . import foundation

_TEST_SIZE = 0.25
_RANDOM_STATE = 0

# Supervised learning on a handful of rows overfits instantly and reports
# meaningless scores; below this we refuse to train rather than hand back a model
# whose metrics can't be trusted.
_MIN_TRAIN_ROWS = 30

_TRAIN_CAVEATS = (
    "Metrics here reflect the training/validation setup — real-world performance "
    "on new data can be lower.",
    "With few rows a model easily overfits — treat scores cautiously.",
)

_FOUNDATION_CAVEAT = (
    f"Predictions come from {foundation.MODEL_LABEL}, a pre-trained tabular foundation "
    "model conditioned on the training rows (no per-task training). It is evaluated "
    "on the same held-out split as any other model."
)

_DEFAULT_BACKEND = "gbt"
_BACKENDS = ("auto", "gbt", "tabicl")
# Backends whose estimator is a foundation model (non-tree, torch-backed).
FOUNDATION_BACKENDS = ("tabicl", "tabicl_finetuned")


def _make_estimator(task: str, backend: str) -> Any:
    """Return an unfitted sklearn-compatible estimator for (task, backend)."""
    if backend == "gbt":
        return (
            HistGradientBoostingClassifier(random_state=_RANDOM_STATE)
            if task == "classification"
            else HistGradientBoostingRegressor(random_state=_RANDOM_STATE)
        )
    if backend == "tabicl":
        return foundation.make_estimator(task)
    raise ValueError(f"Unknown backend {backend!r}; choose one of {_BACKENDS}.")


class TrainedModel:
    """A callable artifact bundling fitted preprocessing + a fitted estimator.

    Unlike Result, this is a live object with behaviour. It also retains the
    held-out test split so evaluate() and permutation importance run on data the
    model never saw during training.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        numeric_features: list[str],
        target: str,
        task: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        nominal_features: list[str] | None = None,
        ordinal_features: list[str] | None = None,
        categorical_features: list[str] | None = None,  # legacy pickled models
        backend: str = "gbt",
        trust: honesty.Trust | None = None,
        selection: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._numeric_features = numeric_features
        self._nominal_features = nominal_features or []
        self._ordinal_features = ordinal_features or []
        # Legacy models pickled before the nominal/ordinal split carried a single
        # categorical_features list; reconstruct nominal from it so old models
        # still unpickle and explain. New models pass nominal/ordinal instead.
        if categorical_features is not None and not self._nominal_features:
            self._nominal_features = categorical_features
        self._categorical_features = self._nominal_features + self._ordinal_features
        self._feature_names = numeric_features + self._categorical_features
        self._target = target
        self._task = task  # "classification" | "regression"
        self._backend = backend  # "gbt" | "tabicl" | "tabicl_finetuned"
        self._X_test = X_test
        self._y_test = y_test
        # Honesty seam — the model's own trust, carried onto the metrics it produces.
        self._trust = trust or honesty.unassessed()
        # How the backend was chosen (requested, chosen, reason, checks) — the
        # deterministic-selection record every downstream result repeats.
        self._selection = selection or {"requested": backend, "chosen": backend}

    @property
    def is_foundation(self) -> bool:
        return getattr(self, "_backend", "gbt") in FOUNDATION_BACKENDS

    def predict(self, X: Any) -> np.ndarray:
        """Predict values / class labels for new rows.

        Args:
            X: A pandas DataFrame, or a dict / list of dicts of feature values.

        Returns:
            Array of predictions.
        """
        with foundation.quiet():
            return self._pipeline.predict(self._as_frame(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities (classifiers only)."""
        if self._task != "classification":
            raise ValueError("predict_proba is only available for classifiers.")
        with foundation.quiet():
            return self._pipeline.predict_proba(self._as_frame(X))

    def _as_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X
        elif isinstance(X, dict):
            frame = pd.DataFrame([X])
        else:
            frame = pd.DataFrame(list(X))
        # Keep only known features; the ColumnTransformer selects by name.
        missing = [c for c in self._feature_names if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns at predict time: {missing}")
        return frame


def train_classifier(store: Any, target: str, backend: str = _DEFAULT_BACKEND) -> TrainedModel:
    """Train a classifier on the table.

    Args:
        store: The Store instance.
        target: Target column name.
        backend: ``"gbt"`` (default, gradient-boosted trees), ``"auto"``
            (deterministic rule) or ``"tabicl"`` (TabICL v2 foundation model).
    """
    return _train(store, target, task="classification", backend=backend)


def train_regressor(store: Any, target: str, backend: str = _DEFAULT_BACKEND) -> TrainedModel:
    """Train a regressor on the table.

    Args:
        store: The Store instance.
        target: Target column name.
        backend: ``"gbt"`` (default, gradient-boosted trees), ``"auto"``
            (deterministic rule) or ``"tabicl"`` (TabICL v2 foundation model).
    """
    return _train(store, target, task="regression", backend=backend)


def _train_declined(
    target: str, task: str, backend: str, n_rows: int, reason: str,
    selection: dict[str, Any] | None = None,
) -> Result:
    """A refusal — the data (or the install) can't support the request, so no model."""
    metadata = {"target": target, "task": task, "backend": backend, "n_rows": int(n_rows)}
    if selection is not None:
        metadata["backend_selection"] = selection
    return Result(
        method="train_declined",
        summary=f"Declined: {reason}",
        values={},
        metadata=metadata,
        trust=honesty.decline(reason, caveats=_TRAIN_CAVEATS, basis=[f"n={n_rows}"]),
    )


def _prepare(store: Any, target: str, task: str, backend: str):
    declined = lambda n, reason: _train_declined(
        target, task, backend, n, reason,
        {"requested": backend, "chosen": None,
         "reason": "Declined before choosing a backend — the data can't support training."},
    )
    """Shared validation for every training path.

    Returns ``(X, y, numeric, nominal, ordinal)``, or a declined Result when the
    data can't support supervised learning.
    """
    frame = store.get_frame()
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} not in table.")

    numeric, nominal, ordinal = _prep.feature_columns(store, exclude=(target,))
    categorical = nominal + ordinal
    if not numeric and not categorical:
        raise ValueError("No usable feature columns to train on.")

    X = frame[numeric + categorical]
    y = frame[target]
    # Drop rows with a missing target for BOTH tasks: neither classifier nor
    # regressor can learn from (or, for the regressor, even accept) a NaN label.
    rows = y.notna()
    X, y = X[rows], y[rows]
    if y.empty:
        raise ValueError(f"Target column {target!r} has no non-null values to train on.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError(f"train_regressor needs a numeric target; {target!r} is not numeric.")

    # Honesty seam — refuse to train when the data can't support supervised
    # learning. A model with meaningless metrics is worse than an honest refusal.
    n_rows = len(X)
    if n_rows < _MIN_TRAIN_ROWS:
        return declined(
            n_rows,
            f"Only {n_rows} usable rows — far too few to train a model that generalises "
            f"(need at least {_MIN_TRAIN_ROWS}); scores would just reflect overfitting.",
        )
    if task == "classification":
        class_counts = y.value_counts()
        if class_counts.size < 2:
            return declined(
                n_rows,
                f"Target {target!r} has only one class in the data — there is nothing to "
                "distinguish, so a classifier cannot learn anything.",
            )
        if int(class_counts.min()) < 2:
            rare = class_counts.idxmin()
            return declined(
                n_rows,
                f"Class {rare!r} appears only once — too few examples to both train on and "
                "hold out, so the model can't learn or be evaluated for that class.",
            )
    return X, y, numeric, nominal, ordinal


def _encoded_width(X: pd.DataFrame, numeric: list[str], nominal: list[str], ordinal: list[str]) -> int:
    """Feature count after one-hot encoding — what the estimator actually sees."""
    return len(numeric) + len(ordinal) + int(sum(X[c].nunique(dropna=True) for c in nominal))


def _split(X: pd.DataFrame, y: pd.Series, task: str):
    """The one held-out split every model of the family is scored on."""
    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    return train_test_split(X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE, stratify=stratify)


def _train(store: Any, target: str, task: str, backend: str = _DEFAULT_BACKEND):
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose one of {_BACKENDS}.")

    prepared = _prepare(store, target, task, backend)
    if isinstance(prepared, Result):
        return prepared
    X, y, numeric, nominal, ordinal = prepared
    n_rows = len(X)
    n_features = _encoded_width(X, numeric, nominal, ordinal)
    n_classes = int(y.nunique()) if task == "classification" else None

    # Method selection — derived from the install and the table's shape only, and
    # recorded so the result says which model answered and why.
    selection = foundation.select_backend(
        backend, task=task, n_rows=n_rows, n_features=n_features, n_classes=n_classes
    )
    chosen = selection["chosen"]
    if chosen == "tabicl":
        # Only an explicit request can land here without the extra or out of
        # envelope ("auto" already checked) — refuse honestly, don't traceback.
        if not foundation.available():
            return _train_declined(
                target, task, chosen, n_rows,
                f"backend='tabicl' requested, but {foundation.INSTALL_HINT}",
                selection,
            )
        problem = foundation.limit_problem(n_rows, n_features)
        if problem:
            return _train_declined(target, task, chosen, n_rows, problem, selection)

    X_train, X_test, y_train, y_test = _split(X, y, task)
    # Neither trees nor TabICL need scaling; still impute + one-hot for uniform
    # handling and a fully numeric matrix the foundation model can consume.
    pre = _prep.build_preprocessor(numeric, nominal, ordinal, scale=False)
    pipeline = Pipeline([("pre", pre), ("model", _make_estimator(task, chosen))])
    try:
        with foundation.quiet():
            pipeline.fit(X_train, y_train)
            if chosen in FOUNDATION_BACKENDS:
                # TabICL's fit only loads weights; the forward pass — where memory
                # or runtime failures surface — happens at predict time. Probe it
                # here so those failures reach the fallback/decline below.
                pipeline.predict(X_test.iloc[:5])
    except Exception as exc:
        if chosen not in FOUNDATION_BACKENDS:
            raise
        # Missing weights, no network, an out-of-memory forward pass: an explicit
        # request declines; "auto" falls back to trees and says so.
        failure = foundation.describe_failure(exc)
        if selection["requested"] != "auto":
            return _train_declined(target, task, chosen, n_rows, failure, selection)
        selection = {
            **selection, "chosen": "gbt", "foundation_error": failure,
            "reason": f"{foundation.MODEL_LABEL} could not run, so fell back to "
                      f"gradient-boosted trees. {failure}",
        }
        chosen = "gbt"
        pipeline = Pipeline([("pre", clone(pre)), ("model", _make_estimator(task, chosen))])
        pipeline.fit(X_train, y_train)

    # Row count drives the floor; a small n can never earn 'high' here (overfit risk).
    caveats = list(_TRAIN_CAVEATS)
    if chosen in FOUNDATION_BACKENDS:
        caveats.append(_FOUNDATION_CAVEAT)
    if "foundation_error" in selection:
        caveats.append(f"The foundation model was unavailable: {selection['foundation_error']}")
    trust = honesty.with_caveats(
        honesty.from_sample_size(n_rows, low=_MIN_TRAIN_ROWS, moderate=200, label="rows"),
        *caveats,
    )

    return TrainedModel(
        pipeline=pipeline,
        numeric_features=numeric,
        nominal_features=nominal,
        ordinal_features=ordinal,
        target=target,
        task=task,
        X_test=X_test,
        y_test=y_test,
        backend=chosen,
        trust=trust,
        selection=selection,
    )


def finetune(
    store: Any,
    target: str,
    task: str,
    max_seconds: float = foundation.FINETUNE_DEFAULT_SECONDS,
):
    """Fine-tune the foundation model's weights on this table, under a time budget.

    In-context use (``backend="tabicl"``) never changes the model; fine-tuning runs
    gradient steps on the training split so the weights adapt to this table. It
    can help on tables unlike the model's synthetic pre-training data, and can
    also overfit — so the pre-trained model is scored on the SAME held-out split
    and the comparison is reported. Guardrails: the foundation model must be
    installed (``install_foundation_model``), at most ``FINETUNE_MAX_ROWS`` rows, and a hard time limit (default
    ``FINETUNE_DEFAULT_SECONDS``, capped at ``FINETUNE_MAX_SECONDS``).

    Args:
        store: The Store instance.
        target: Target column name.
        task: ``"classification"`` or ``"regression"``.
        max_seconds: Wall-clock budget for the gradient steps.

    Returns:
        A TrainedModel (backend ``"tabicl_finetuned"``), or a declined Result.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'.")
    backend = "tabicl_finetuned"
    prepared = _prepare(store, target, task, backend)
    if isinstance(prepared, Result):
        return prepared
    X, y, numeric, nominal, ordinal = prepared
    n_rows = len(X)
    n_features = _encoded_width(X, numeric, nominal, ordinal)
    budget = float(min(max(max_seconds, 10), foundation.FINETUNE_MAX_SECONDS))
    selection: dict[str, Any] = {
        "requested": "finetune", "chosen": backend,
        "reason": f"Fine-tuning {foundation.MODEL_LABEL} requested explicitly.",
        "checks": {"foundation_installed": foundation.available(), "n_rows": n_rows,
                   "n_encoded_features": n_features,
                   "finetune_max_rows": foundation.FINETUNE_MAX_ROWS},
    }

    if not foundation.available():
        return _train_declined(
            target, task, backend, n_rows,
            f"Fine-tuning requested, but {foundation.INSTALL_HINT}",
            selection,
        )
    if n_rows > foundation.FINETUNE_MAX_ROWS:
        return _train_declined(
            target, task, backend, n_rows,
            f"{n_rows:,} rows is too many to fine-tune on a CPU in reasonable time (limit "
            f"{foundation.FINETUNE_MAX_ROWS:,}). Use train_* with backend='tabicl' — the "
            "pre-trained model needs no fine-tuning to work.",
            selection,
        )
    problem = foundation.limit_problem(n_rows, n_features)
    if problem:
        return _train_declined(target, task, backend, n_rows, problem, selection)

    X_train, X_test, y_train, y_test = _split(X, y, task)
    pre = _prep.build_preprocessor(numeric, nominal, ordinal, scale=False)
    pretrained = Pipeline([("pre", pre), ("model", foundation.make_estimator(task))])
    tuned = Pipeline([("pre", clone(pre)), ("model", foundation.make_finetuner(task, budget))])
    try:
        with foundation.quiet():
            pretrained.fit(X_train, y_train)
            started = time.monotonic()
            tuned.fit(X_train, y_train)
            elapsed = time.monotonic() - started
            score_pre = _primary_score(task, y_test, pretrained.predict(X_test))
            score_tuned = _primary_score(task, y_test, tuned.predict(X_test))
    except Exception as exc:
        return _train_declined(target, task, backend, n_rows,
                               foundation.describe_failure(exc), selection)

    metric = "accuracy" if task == "classification" else "r2"
    improved = score_tuned > score_pre
    # tabicl stops when the NEXT epoch would overrun, so a truncated run can end
    # well short of the budget; a loose threshold errs toward the caveat.
    hit_budget = elapsed >= budget * 0.6
    selection["finetune"] = {
        "max_epochs": foundation.FINETUNE_EPOCHS, "time_limit_s": budget,
        "elapsed_s": round(elapsed, 1), "stopped_by_time_limit": hit_budget,
        "held_out_metric": metric,
        "held_out_pretrained": score_pre, "held_out_finetuned": score_tuned,
        "improved_over_pretrained": improved,
    }

    caveats = [*_TRAIN_CAVEATS, _FOUNDATION_CAVEAT,
               f"Fine-tuned on this table: held-out {metric} {score_tuned:.3f} vs "
               f"{score_pre:.3f} for the pre-trained model on the same split."]
    if not improved:
        caveats.append(
            "Fine-tuning did NOT beat the pre-trained model on held-out data — prefer "
            "train_* with backend='tabicl' for this table."
        )
    if hit_budget:
        caveats.append(
            "Fine-tuning stopped at its time limit, so how many epochs ran depends on "
            "machine speed — a rerun elsewhere can give slightly different weights."
        )
    trust = honesty.with_caveats(
        honesty.from_sample_size(n_rows, low=_MIN_TRAIN_ROWS, moderate=200, label="rows"),
        *caveats,
    )
    return TrainedModel(
        pipeline=tuned,
        numeric_features=numeric,
        nominal_features=nominal,
        ordinal_features=ordinal,
        target=target,
        task=task,
        X_test=X_test,
        y_test=y_test,
        backend=backend,
        trust=trust,
        selection=selection,
    )


def _primary_score(task: str, y_true: Any, y_pred: Any) -> float:
    """The headline held-out score used to compare two models of one task."""
    if task == "classification":
        return float(accuracy_score(y_true, y_pred))
    return float(r2_score(y_true, y_pred))


def evaluate(store: Any, model: TrainedModel) -> Result:
    """Evaluate a trained model on its held-out test split.

    Classifiers: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and
    the majority-class baseline accuracy. Regressors: MAE, RMSE, R².

    Args:
        store: The Store instance (kept for signature symmetry; the held-out
            split lives on the model).
        model: A TrainedModel from train_classifier / train_regressor.

    Returns:
        Result with the metric set and evaluation parameters.
    """
    X_test, y_test = model._X_test, model._y_test
    y_pred = model.predict(X_test)

    if model._task == "classification":
        values: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            # What "always predict the most common class" would score on this split.
            "baseline_accuracy": float(pd.Series(y_test).value_counts(normalize=True).iloc[0]),
        }
        values["roc_auc"] = _safe_roc_auc(model, X_test, y_test)
        summary = f"accuracy={values['accuracy']:.3f}, f1={values['f1']:.3f}"
        method = "classification_metrics"
        no_skill = values["accuracy"] <= values["baseline_accuracy"]
        no_skill_note = (
            f"The model is no better than always predicting the most common class "
            f"(accuracy {values['accuracy']:.3f} vs baseline {values['baseline_accuracy']:.3f})."
        )
    else:
        values = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(root_mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        summary = f"R²={values['r2']:.3f}, RMSE={values['rmse']:.3g}"
        method = "regression_metrics"
        no_skill = values["r2"] <= 0
        no_skill_note = (
            f"R²={values['r2']:.3f} — the model is no better than predicting the mean."
        )

    # Honesty seam — trust from the held-out sample size, folded together with the
    # model's own training trust; the metric set only reflects THIS test split.
    n_test = int(len(y_test))
    eval_trust = honesty.with_caveats(
        honesty.from_sample_size(n_test, low=30, moderate=100, label="test rows"),
        "These scores come from one held-out split — a different split or fresh data "
        "can score differently.",
    )
    parts = [getattr(model, "_trust", None), eval_trust]
    if no_skill:
        # A model with no held-out skill must not read as trustworthy however
        # many rows it saw.
        parts.append(honesty.Trust(level=honesty.TrustLevel.LOW, caveats=[no_skill_note],
                                   basis=["no held-out skill"]))
    trust = honesty.combine(*parts)

    backend = getattr(model, "_backend", "gbt")
    return Result(
        method=method,
        summary=summary,
        values=values,
        metadata={
            "target": model._target,
            "task": model._task,
            "backend": backend,
            "backend_selection": getattr(model, "_selection", {"chosen": backend}),
            "n_test": n_test,
        },
        trust=trust,
    )


def _safe_roc_auc(model: TrainedModel, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """ROC-AUC, handling binary vs multiclass; None if it can't be computed."""
    try:
        proba = model.predict_proba(X_test)
        classes = model._pipeline.named_steps["model"].classes_
        if len(classes) == 2:
            return float(roc_auc_score(y_test, proba[:, 1]))
        return float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
    except Exception:
        return None
