"""The honesty seam — every result says how much to trust it, and can decline.

Capability you can't verify is a trust product: the user pays *because* they
can't do this themselves, so they also can't check the number. Every ``Result``
therefore carries a ``Trust`` — a confidence level, the caveats behind it, and,
when the data can't support the question, a *refusal to answer* instead of a
misleading number.

Two properties, by design:
  * **Uniform.** ``_serialize.result_dict`` always emits a ``trust`` block, so
    all 44 tools have the shape from day one (``unassessed`` until an analytic
    populates it) — no confident-number-by-default.
  * **Ergonomic.** Analytics attach real confidence in a line or two via the
    assessors here (``from_sample_size``, ``combine``, ``with_caveats``) or refuse
    via ``decline``. Installed while stakes are low so it's habitual by the time a
    wrong causal answer costs real money.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field


class TrustLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"            # paired with declined=True
    UNASSESSED = "unassessed"  # explicit "not yet judged" — never a fake 'high'


# Ordering for "most cautious wins" when combining. UNASSESSED sorts as cautious.
_ORDER = {
    TrustLevel.NONE: 0,
    TrustLevel.UNASSESSED: 1,
    TrustLevel.LOW: 2,
    TrustLevel.MODERATE: 3,
    TrustLevel.HIGH: 4,
}


class Trust(BaseModel):
    """How much to trust a result, and why."""

    level: TrustLevel = TrustLevel.UNASSESSED
    caveats: list[str] = Field(default_factory=list)   # plain-language "read this carefully"
    basis: list[str] = Field(default_factory=list)     # what drove the level, e.g. "n=42"
    declined: bool = False
    decline_reason: str | None = None


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def unassessed(note: str | None = None) -> Trust:
    return Trust(
        level=TrustLevel.UNASSESSED,
        caveats=[note or "Confidence not yet assessed for this method — interpret with care."],
    )


def decline(
    reason: str,
    *,
    caveats: Iterable[str] | None = None,
    basis: Iterable[str] | None = None,
) -> Trust:
    """Refuse to answer: the data can't support the question. A refusal is a
    stronger trust signal than a meaningless number."""
    return Trust(
        level=TrustLevel.NONE,
        declined=True,
        decline_reason=reason,
        caveats=list(caveats or []),
        basis=list(basis or []),
    )


def from_sample_size(
    n: int, *, low: int = 30, moderate: int = 100, label: str = "observations"
) -> Trust:
    """Coarse data-sufficiency signal: ``n<low`` → low, ``<moderate`` → moderate,
    else high. A floor, not the last word — specific methods add their own caveats."""
    if n < low:
        return Trust(
            level=TrustLevel.LOW,
            caveats=[f"Small sample ({n} {label}) — estimates are noisy; treat as directional."],
            basis=[f"n={n}"],
        )
    if n < moderate:
        return Trust(
            level=TrustLevel.MODERATE,
            caveats=[f"Moderate sample ({n} {label}) — reasonable but not definitive."],
            basis=[f"n={n}"],
        )
    return Trust(level=TrustLevel.HIGH, basis=[f"n={n}"])


def combine(*trusts: Trust) -> Trust:
    """Merge trusts: the most cautious level wins; caveats/basis are unioned; a
    single decline dominates."""
    present = [t for t in trusts if t is not None]
    if not present:
        return unassessed()
    declined = [t for t in present if t.declined]
    caveats = _dedupe(c for t in present for c in t.caveats)
    basis = _dedupe(b for t in present for b in t.basis)
    if declined:
        return Trust(
            level=TrustLevel.NONE, declined=True,
            decline_reason=declined[0].decline_reason, caveats=caveats, basis=basis,
        )
    level = min((TrustLevel(t.level) for t in present), key=lambda lv: _ORDER[lv])
    return Trust(level=level, caveats=caveats, basis=basis)


def with_caveats(trust: Trust, *caveats: str) -> Trust:
    return trust.model_copy(update={"caveats": _dedupe([*trust.caveats, *caveats])})
