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

* ``available()`` — are the packages AND weights installed (no heavy import)?
* ``select_backend()`` — the deterministic ``backend="auto"`` rule, recorded.
* ``make_estimator()`` / ``make_finetuner()`` — pinned checkpoint, CPU, seed.
* ``quiet()`` — tabicl ``print()``s download notices; on a stdio transport
  stdout IS the JSON-RPC stream, so every call into it is redirected to stderr.
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

# The automatic rule only picks the foundation model inside the envelope where it
# is both strong and quick on a CPU (~5s to fit and score 1k rows on a laptop).
# Explicit backend="tabicl" may go further, up to the model's stated ceilings.
AUTO_MAX_ROWS = 10_000
AUTO_MAX_FEATURES = 500
AUTO_MAX_CLASSES = 10
MAX_ROWS = 100_000
MAX_FEATURES = 2_000

# Fine-tuning runs real gradient steps; on a CPU that is only reasonable on small
# tables and inside a hard time budget.
FINETUNE_MAX_ROWS = 5_000
FINETUNE_DEFAULT_SECONDS = 120
FINETUNE_MAX_SECONDS = 600
FINETUNE_EPOCHS = 10

INSTALL_HINT = (
    f"The {MODEL_LABEL} foundation model is not installed. Call install_foundation_model "
    "to install it in the background (one-time: PyTorch CPU + ~220 MB of weights, a few "
    "minutes), call it again until it reports state 'ready', then retrain."
)

# Opt-in: start the background install the first time "auto" would have used it.
_AUTO_INSTALL_ENV = "MEELU_FOUNDATION_AUTO_INSTALL"


def available() -> bool:
    """True when the packages are importable AND the weights are cached, so a
    training call never starts a large download. Uses find_spec and a cache
    lookup — never pays torch's import cost just to answer the question."""
    return foundation_install.installed()


def device() -> str:
    """Torch device for the foundation model. CPU by default — results must be
    reproducible on any machine; MEELU_FOUNDATION_DEVICE=cuda opts into a GPU."""
    return os.environ.get("MEELU_FOUNDATION_DEVICE", "cpu")


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Send anything the foundation library prints to stderr, not stdout."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


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
        "n_rows": int(n_rows),
        "n_encoded_features": int(n_features),
        "n_classes": None if n_classes is None else int(n_classes),
        "auto_limits": {"rows": AUTO_MAX_ROWS, "features": AUTO_MAX_FEATURES,
                        "classes": AUTO_MAX_CLASSES},
    }
    if requested != "auto":
        return {"requested": requested, "chosen": requested,
                "reason": f"backend={requested!r} requested explicitly.", "checks": checks}

    in_envelope = _envelope_problem(task, n_rows, n_features, n_classes) is None
    if not checks["foundation_installed"]:
        reason = f"{MODEL_LABEL} foundation model not installed — using gradient-boosted trees."
        if not in_envelope:
            reason += " " + _envelope_problem(task, n_rows, n_features, n_classes)
            return {"requested": "auto", "chosen": "gbt", "reason": reason, "checks": checks}
        out = {"requested": "auto", "chosen": "gbt", "reason": reason, "checks": checks,
               "hint": INSTALL_HINT}
        if os.environ.get(_AUTO_INSTALL_ENV) == "1":
            state = foundation_install.start()["state"]
            out["hint"] = (f"{_AUTO_INSTALL_ENV}=1: background install is {state}; check "
                           "it with install_foundation_model and retrain once it is 'ready'.")
        return out
    problem = _envelope_problem(task, n_rows, n_features, n_classes)
    if problem is None:
        return {"requested": "auto", "chosen": "tabicl",
                "reason": (f"{MODEL_LABEL} installed and the table is within its "
                           f"envelope (≤{AUTO_MAX_ROWS:,} rows, ≤{AUTO_MAX_FEATURES} "
                           f"features, ≤{AUTO_MAX_CLASSES} classes) — using {MODEL_LABEL}."),
                "checks": checks}
    return {"requested": "auto", "chosen": "gbt", "reason": problem, "checks": checks}


def _envelope_problem(task: str, n_rows: int, n_features: int, n_classes: int | None) -> str | None:
    """Why ``auto`` should not use the foundation model on this shape, or None."""
    if n_rows > AUTO_MAX_ROWS:
        return (f"{n_rows:,} rows exceeds the foundation model's CPU budget "
                f"({AUTO_MAX_ROWS:,}) — using gradient-boosted trees.")
    if n_features > AUTO_MAX_FEATURES:
        return (f"{n_features:,} encoded features exceeds {AUTO_MAX_FEATURES:,} — "
                "using gradient-boosted trees.")
    if task == "classification" and n_classes is not None and n_classes > AUTO_MAX_CLASSES:
        return (f"{n_classes} classes exceeds {AUTO_MAX_CLASSES} — "
                "using gradient-boosted trees.")
    return None


def limit_problem(n_rows: int, n_features: int) -> str | None:
    """Why an explicit foundation request can't run on this table, or None."""
    if n_rows > MAX_ROWS:
        return (f"{MODEL_LABEL} supports up to ~{MAX_ROWS:,} rows on this server; the table "
                f"has {n_rows:,}. Use backend='gbt' for larger tables.")
    if n_features > MAX_FEATURES:
        return (f"{MODEL_LABEL} supports up to ~{MAX_FEATURES:,} features; the table has "
                f"{n_features:,} after encoding. Use backend='gbt' instead.")
    return None


def make_estimator(task: str) -> Any:
    """Unfitted TabICL estimator: pinned checkpoint, fixed seed, explicit device.

    No KV cache: it makes repeated predictions faster but pickles to hundreds of
    MB, and every model here is persisted with the session. The pickle without it
    is ~200 KB — the weights reload from the local Hugging Face cache.
    """
    from tabicl import TabICLClassifier, TabICLRegressor

    if task == "classification":
        return TabICLClassifier(checkpoint_version=CLASSIFIER_CHECKPOINT,
                                device=device(), random_state=_RANDOM_STATE)
    return TabICLRegressor(checkpoint_version=REGRESSOR_CHECKPOINT,
                           device=device(), random_state=_RANDOM_STATE)


def make_finetuner(task: str, time_limit: float) -> Any:
    """Unfitted fine-tuning estimator with CPU-safe settings and a hard time limit."""
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
                                   "huggingface", "hf_hub", "localentrynotfound", "not cached")):
        return (f"Could not load the {MODEL_LABEL} weights from the local cache. Call "
                f"install_foundation_model to re-download them. ({text[:200]})")
    return f"The {MODEL_LABEL} model failed on this table ({text[:300]})."

