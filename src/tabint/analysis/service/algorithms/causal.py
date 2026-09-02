"""Causal effect estimation — "what actually moves the metric", not just correlation.

``causal_effect`` estimates the average effect of a ``treatment`` column on an
``outcome`` column, adjusting for confounders via the backdoor criterion, and then
runs a refutation test so the number comes with a credibility check.

Library: **DoWhy** (``CausalModel``: identify → estimate → refute), imported lazily
via the optional ``insights`` extra. DoWhy encodes the whole identify/estimate/refute
workflow; we only choose sensible defaults (all other usable columns as common causes)
and format the result.
"""
from __future__ import annotations

from typing import Any

from ....shared import honesty
from ....shared.results import Result
from .. import _prep

# Observational causal inference is the easiest thing in data science to get
# confidently wrong, so the bar to answer at all is deliberately high.
_MIN_ROWS = 50
_BASE_CAVEATS = (
    "Estimated from observational data, not an experiment: this rests on the backdoor "
    "assumption that the adjusted confounders capture the relevant common causes.",
    "Unmeasured confounding or reverse causality could bias this — treat it as a hypothesis "
    "to test (ideally with an experiment), not a proven cause.",
)


def _declined(treatment, outcome, confounders, n_rows, reason) -> Result:
    """A refusal — no headline effect number, a clear reason, base caveats."""
    return Result(
        method="causal_effect_declined",
        summary=f"Declined: {reason}",
        values={},
        metadata={
            "treatment": treatment, "outcome": outcome,
            "confounders": confounders, "n_rows": int(n_rows),
        },
        trust=honesty.decline(reason, caveats=_BASE_CAVEATS, basis=[f"n={n_rows}"]),
    )


def causal_effect(
    store: Any,
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
) -> Result:
    """Estimate the average causal effect of ``treatment`` on ``outcome``.

    Args:
        store: The Store/Table instance.
        treatment: The intervention column (binary or continuous).
        outcome: The numeric outcome column.
        confounders: Columns to adjust for. Defaults to every other usable feature
            column (a backdoor-adjustment starting point — refine per domain).

    Returns:
        Result with ``effect`` (point estimate), ``refutation`` (placebo/random-cause
        check), and the confounder set used.
    """
    frame = store.get_frame()
    for col in (treatment, outcome):
        if col not in frame.columns:
            raise ValueError(f"Column {col!r} not in table.")

    if confounders is None:
        numeric, nominal, ordinal = _prep.feature_columns(store, exclude=(treatment, outcome))
        confounders = numeric + nominal + ordinal
    else:
        missing = [c for c in confounders if c not in frame.columns]
        if missing:
            raise ValueError(f"Confounder columns not in table: {missing}")

    data = frame[[treatment, outcome, *confounders]].dropna()
    if data.empty:
        raise ValueError("No complete rows across treatment, outcome, and confounders.")

    # Honesty seam — decline BEFORE touching DoWhy when the data can't support a
    # causal claim. A meaningless effect number is worse than an honest refusal.
    if len(data) < _MIN_ROWS:
        return _declined(
            treatment, outcome, confounders, len(data),
            f"Only {len(data)} complete rows across treatment, outcome and confounders — "
            f"far too few to support a credible causal estimate (need at least {_MIN_ROWS}).",
        )
    if data[treatment].nunique() < 2:
        return _declined(
            treatment, outcome, confounders, len(data),
            f"{treatment!r} takes only one value in the data — there is no counterfactual to "
            "compare against, so no causal effect is identifiable.",
        )

    try:
        from dowhy import CausalModel
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "causal_effect needs DoWhy. Install it with:  pip install 'tabint[insights]'."
        ) from exc

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders or None,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified, method_name="backdoor.linear_regression"
    )
    effect = float(estimate.value)

    # Refutation: adding a random common cause should NOT move a real effect much.
    refutation: dict[str, Any]
    try:
        ref = model.refute_estimate(
            identified, estimate, method_name="random_common_cause"
        )
        refutation = {
            "method": "random_common_cause",
            "new_effect": float(ref.new_effect)
            if isinstance(ref.new_effect, (int, float))
            else None,
            "passed": abs((ref.new_effect or 0) - effect) < 0.5 * (abs(effect) + 1e-9)
            if isinstance(ref.new_effect, (int, float))
            else None,
        }
    except Exception as exc:  # refutation is best-effort, never fatal
        refutation = {"method": "random_common_cause", "error": str(exc)}

    # Honesty seam — a failed placebo refutation means the estimate is unreliable;
    # withhold the number rather than present it as an answer.
    passed = refutation.get("passed")
    if passed is False:
        return Result(
            method="causal_effect_declined",
            summary=(
                f"Declined: the estimate for {treatment!r} → {outcome!r} failed a placebo "
                "refutation (a random common cause shifted it substantially), so the data "
                "cannot support a trustworthy causal claim here."
            ),
            values={"refutation": refutation},  # diagnostics only — NOT a headline effect
            metadata={
                "treatment": treatment, "outcome": outcome, "confounders": confounders,
                "n_rows": int(len(data)), "estimand": "backdoor", "effect_withheld": effect,
            },
            trust=honesty.decline(
                "Failed a placebo refutation test — the estimate is not robust.",
                caveats=_BASE_CAVEATS, basis=[f"n={len(data)}", "refutation_passed=False"],
            ),
        )

    # Observational causal never earns 'high'. Cap at moderate, and only with a
    # large sample and a passed refutation; otherwise low.
    level = (
        honesty.TrustLevel.MODERATE
        if (len(data) >= 500 and passed)
        else honesty.TrustLevel.LOW
    )
    caveats = list(_BASE_CAVEATS)
    if passed is None:
        caveats.append(
            "Refutation could not be computed, so the estimate's robustness is unverified."
        )
    trust = honesty.Trust(
        level=level, caveats=caveats, basis=[f"n={len(data)}", f"refutation_passed={passed}"]
    )
    return Result(
        method="dowhy_backdoor_linear_regression",
        summary=f"Estimated effect of {treatment!r} on {outcome!r}: {effect:.4g}",
        values={"effect": effect, "refutation": refutation},
        metadata={
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "n_rows": int(len(data)),
            "estimand": "backdoor",
        },
        trust=trust,
    )
