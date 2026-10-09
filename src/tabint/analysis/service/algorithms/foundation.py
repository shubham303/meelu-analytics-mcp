"""Tabular foundation model — TabICL v2, installed on demand.

A foundation model is a transformer pre-trained on millions of synthetic tables.
Given the training rows as *context* it predicts new rows in one forward pass —
no per-task gradient training — and is usually at least as accurate as tuned
gradient boosting on small and medium tables. That makes it the natural default
for "which of these customers will churn?" questions on a few thousand rows.

Why TabICL and not TabPFN: both code AND weights are BSD-3-Clause, the weights
download from Hugging Face without a login, and the package is light (torch +
einops + huggingface-hub). Current TabPFN releases default to non-commercial
weights behind a browser license flow that cannot complete inside a stdio MCP
server. See docs/tools/foundation-models.md for the full comparison.

This module owns everything foundation-specific so supervised.py stays a router
(installation itself lives in ``foundation_install``):

* ``available()`` — are the packages AND weights installed (no import, no wiring)?
* ``select_backend()`` — the deterministic ``backend="auto"`` rule, recorded.
* ``make_estimator()`` / ``make_finetuner()`` — pinned checkpoint, CPU, seed,
  downloads disabled; the only places the runtime folder joins ``sys.path``.
* ``quiet()`` — tabicl ``print()``s download notices; on a stdio transport
  stdout IS the JSON-RPC stream, so every call into it is redirected to stderr.

CPU latency budget. Every TabICL predict call re-reads its whole training
context, and on a laptop CPU its cost grows with context rows × features
(measured on an 8 GB Apple-silicon laptop, 4 threads, 8 estimators: ~10 s for
1,500 context rows × 10 features, ~35 s at × 40 features, and only ~40 rows/s of
query throughput). FastMCP runs tools on its event loop, so a slow call freezes
the whole server. The envelope below therefore comes from a latency budget —
every foundation tool call should finish in about a minute on a CPU — not from
the model's accuracy ceiling.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Iterator

from . import foundation_install
from .foundation_install import CLASSIFIER_CHECKPOINT, REGRESSOR_CHECKPOINT

MODEL_LABEL = "TabICL v2"

_RANDOM_STATE = 0

# Ensemble size. TabICL's default is 8; on a CPU 4 halves every call's cost for a
# small accuracy loss. Importance/SHAP rescore many perturbed copies, so they use
# a 2-member refit of the same model (see interpretation.py).
N_ESTIMATORS_CPU = 4
N_ESTIMATORS_GPU = 8
N_ESTIMATORS_EXPLAIN = 2

# The envelope (CPU). Rows are usable training rows (75% become the context);
# "cells" = rows × encoded features, which is what per-call cost tracks. At the
# edge: train + held-out scoring ≈ 20 s, add_predictions on the whole table
# ≈ 30 s. The same limits bound explicit backend="tabicl" — beyond them every
# call would take minutes. A GPU (MEELU_FOUNDATION_DEVICE=cuda) lifts them.
AUTO_MAX_ROWS = 2_000
AUTO_MAX_CELLS = 30_000
AUTO_MAX_CLASSES = 10
_GPU_SCALE = 25
# Most rows add_predictions will score in one call (the table may hold rows the
# model didn't train on, e.g. missing targets).
PREDICT_MAX_ROWS = 4_000

# Fine-tuning runs real gradient steps plus two held-out scorings; on a CPU that
# fits the one-minute budget only on small tables and a short time limit.
FINETUNE_MAX_ROWS = 1_000
FINETUNE_DEFAULT_SECONDS = 20
FINETUNE_MAX_SECONDS = 30
FINETUNE_EPOCHS = 10

INSTALL_HINT = (
    f"The {MODEL_LABEL} foundation model is not installed. Call install_foundation_model "
    "to install it in the background (one-time: PyTorch CPU + ~220 MB of weights, a few "
    "minutes), call it again until it reports state 'ready', then retrain."
)

# Opt-in: start the background install the first time "auto" would have used it.
_AUTO_INSTALL_ENV = "MEELU_FOUNDATION_AUTO_INSTALL"


def available() -> bool:
    """True when the packages are present AND the weights are cached, so a
    training call never starts a large download. A stamp + cache lookup — no
    torch import, and the runtime folder is NOT put on sys.path."""
    return foundation_install.installed()


def device() -> str:
    """Torch device for the foundation model. CPU by default — results must be
    reproducible on any machine; MEELU_FOUNDATION_DEVICE=cuda opts into a GPU."""
    return os.environ.get("MEELU_FOUNDATION_DEVICE", "cpu")


def _scale() -> int:
    return 1 if device() == "cpu" else _GPU_SCALE


def max_rows() -> int:
    return AUTO_MAX_ROWS * _scale()


def max_cells() -> int:
    return AUTO_MAX_CELLS * _scale()


def predict_max_rows() -> int:
    return PREDICT_MAX_ROWS * _scale()


def n_estimators() -> int:
    return N_ESTIMATORS_CPU if device() == "cpu" else N_ESTIMATORS_GPU


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Send anything the foundation library prints to stderr, not stdout."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def runtime_info() -> dict[str, Any]:
    """What a foundation result was produced with — device, checkpoints, ensemble
    size and library versions — for the result's metadata."""
    return {
        "device": device(),
        "n_estimators": n_estimators(),
        "checkpoints": {"classification": CLASSIFIER_CHECKPOINT,
                        "regression": REGRESSOR_CHECKPOINT},
        **{f"{k}_version": v for k, v in foundation_install.runtime_versions().items()},
    }


def select_backend(
    requested: str, *, task: str, n_rows: int, n_features: int, n_classes: int | None
) -> dict[str, Any]:
    """Resolve ``backend="auto"`` deterministically and say why.

    The rule depends only on whether the model is installed and on the table's
    shape — both recorded in ``checks`` — so the same table on the same install
    always gets the same choice. The returned dict goes straight into the
    result's metadata. When the model is not installed but the table is within
    its envelope, ``hint`` tells the caller how to get it.
    """
    checks = {
        "foundation_installed": available(),
        "device": device(),
        "n_rows": int(n_rows),
        "n_encoded_features": int(n_features),
        "n_cells": int(n_rows) * int(n_features),
        "n_classes": None if n_classes is None else int(n_classes),
        "auto_limits": {"rows": max_rows(), "cells": max_cells(),
                        "classes": AUTO_MAX_CLASSES},
    }
    if requested != "auto":
        return {"requested": requested, "chosen": requested,
                "reason": f"backend={requested!r} requested explicitly.", "checks": checks}

    problem = _envelope_problem(task, n_rows, n_features, n_classes)
    if not checks["foundation_installed"]:
        reason = f"{MODEL_LABEL} foundation model not installed — using gradient-boosted trees."
        if problem is not None:
            return {"requested": "auto", "chosen": "gbt", "reason": f"{reason} {problem}",
                    "checks": checks}
        out = {"requested": "auto", "chosen": "gbt", "reason": reason, "checks": checks,
               "hint": INSTALL_HINT}
        if os.environ.get(_AUTO_INSTALL_ENV) == "1":
            state = foundation_install.start()["state"]
            out["hint"] = (f"{_AUTO_INSTALL_ENV}=1: background install is {state}; check "
                           "it with install_foundation_model and retrain once it is 'ready'.")
        return out
    if problem is None:
        return {"requested": "auto", "chosen": "tabicl",
                "reason": (f"{MODEL_LABEL} installed and the table is within its CPU "
                           f"envelope (≤{max_rows():,} rows, ≤{max_cells():,} rows×features, "
                           f"≤{AUTO_MAX_CLASSES} classes) — using {MODEL_LABEL}."),
                "checks": checks}
    return {"requested": "auto", "chosen": "gbt", "reason": problem, "checks": checks}


def _envelope_problem(task: str, n_rows: int, n_features: int, n_classes: int | None) -> str | None:
    """Why ``auto`` should not use the foundation model on this shape, or None."""
    if n_rows > max_rows():
        return (f"{n_rows:,} rows exceeds the foundation model's {device().upper()} latency "
                f"budget ({max_rows():,}) — using gradient-boosted trees.")
    if n_rows * n_features > max_cells():
        return (f"{n_rows:,} rows × {n_features:,} encoded features exceeds the foundation "
                f"model's {device().upper()} latency budget ({max_cells():,}) — using "
                "gradient-boosted trees.")
    if task == "classification" and n_classes is not None and n_classes > AUTO_MAX_CLASSES:
        return (f"{n_classes} classes exceeds {AUTO_MAX_CLASSES} — "
                "using gradient-boosted trees.")
    return None


def limit_problem(n_rows: int, n_features: int) -> str | None:
    """Why an explicit foundation request can't run on this table, or None.
    Same latency budget as ``auto`` (classes aside) — past it, every call on a
    CPU would take minutes and freeze the server meanwhile."""
    if n_rows > max_rows() or n_rows * n_features > max_cells():
        return (f"{MODEL_LABEL} on {device().upper()} is limited to {max_rows():,} rows and "
                f"{max_cells():,} rows×features here (this table: {n_rows:,} × {n_features:,}) "
                "— beyond that each call takes minutes. Use backend='gbt', or a GPU via "
                "MEELU_FOUNDATION_DEVICE=cuda.")
    return None


def _load() -> None:
    """Put the on-demand runtime on sys.path — the only time it joins."""
    if not foundation_install.wire():
        raise ImportError(f"{MODEL_LABEL} runtime is not installed. {INSTALL_HINT}")


def make_estimator(task: str, n_estimators_: int | None = None) -> Any:
    """Unfitted TabICL estimator: pinned checkpoint, fixed seed, explicit device,
    downloads disabled (weights come only from install_foundation_model).

    No KV cache: it makes repeated predictions faster but pickles to hundreds of
    MB, and every model here is persisted with the session. The pickle without it
    is ~200 KB — the weights reload from the local Hugging Face cache.
    """
    _load()
    from tabicl import TabICLClassifier, TabICLRegressor

    cls, ckpt = ((TabICLClassifier, CLASSIFIER_CHECKPOINT) if task == "classification"
                 else (TabICLRegressor, REGRESSOR_CHECKPOINT))
    return cls(checkpoint_version=ckpt, n_estimators=n_estimators_ or n_estimators(),
               allow_auto_download=False, device=device(), random_state=_RANDOM_STATE)


def make_finetuner(task: str, time_limit: float) -> Any:
    """Unfitted fine-tuning estimator with CPU-safe settings and a hard time limit."""
    _load()
    from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor

    cls, ckpt = (
        (FinetunedTabICLClassifier, CLASSIFIER_CHECKPOINT)
        if task == "classification"
        else (FinetunedTabICLRegressor, REGRESSOR_CHECKPOINT)
    )
    return cls(
        epochs=FINETUNE_EPOCHS,
        time_limit=float(time_limit),
        early_stopping=True,
        checkpoint_version=ckpt,
        allow_auto_download=False,
        n_estimators_inference=n_estimators(),
        device=device(),
        # Mixed precision is a GPU speed trick; off keeps CPU runs bit-stable.
        amp=False,
        random_state=_RANDOM_STATE,
    )


def describe_failure(exc: BaseException) -> str:
    """Turn a load/fit failure into a sentence a user can act on."""
    text = f"{type(exc).__name__}: {exc}".strip()
    lowered = text.lower()
    if isinstance(exc, ImportError):
        return f"The foundation model's packages could not be imported ({text}). {INSTALL_HINT}"
    if any(k in lowered for k in ("offline", "connection", "resolve", "timed out", "timeout",
                                   "huggingface", "hf_hub", "localentrynotfound", "not cached",
                                   "checkpoint")):
        return (f"Could not load the {MODEL_LABEL} weights from the local cache. Call "
                f"install_foundation_model to re-download them. ({text[:200]})")
    return f"The {MODEL_LABEL} model failed on this table ({text[:300]})."
