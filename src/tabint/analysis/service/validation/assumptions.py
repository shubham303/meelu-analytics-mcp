"""Statistical assumption checks that route test selection in the association family.

These helpers answer questions like "is this column normally distributed?" or
"do these groups have equal variance?" The answers drive which test analyze_association
selects — e.g., Pearson vs. Spearman, t-test vs. Mann-Whitney, chi-square vs. Fisher.

All checks are deterministic and delegate the actual statistics to scipy.stats.
The alpha level for every test is fixed at 0.05.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_ALPHA = 0.05

# Shapiro-Wilk is the more powerful test for small samples but grows
# over-sensitive as n rises — beyond ~50 points it rejects normality on trivial
# deviations. Switch to D'Agostino-Pearson (normaltest) above this size.
_SHAPIRO_MAX_N = 50


def _clean(series: Any) -> np.ndarray:
    """Return the non-null numeric values of a series as a 1-D float array."""
    arr = pd.Series(series).dropna().to_numpy(dtype="float64", na_value=np.nan)
    return arr[~np.isnan(arr)]


def is_normal(series: Any) -> bool:
    """Test whether a series is approximately normally distributed.

    Uses Shapiro-Wilk for small samples and D'Agostino-Pearson (normaltest)
    for larger ones. Returns False when there are too few points to test.

    Args:
        series: A pandas Series (or array-like) of numeric values.

    Returns:
        True if the null hypothesis of normality cannot be rejected at α=0.05.
    """
    x = _clean(series)
    n = x.size
    # Constant data has zero variance — not "normal" in any useful sense.
    if n < 3 or np.ptp(x) == 0:
        return False
    try:
        if n <= _SHAPIRO_MAX_N:
            _, p = stats.shapiro(x)
        else:
            _, p = stats.normaltest(x)
    except Exception:
        return False
    return bool(p > _ALPHA)


def has_equal_variance(*groups: Any) -> bool:
    """Test whether multiple groups have equal variance (homoscedasticity).

    Uses Levene's test (median-centred, robust to non-normality).

    Args:
        *groups: Two or more pandas Series (or array-likes), one per group.

    Returns:
        True if equal variance cannot be rejected at α=0.05.
    """
    cleaned = [_clean(g) for g in groups]
    cleaned = [g for g in cleaned if g.size >= 2]
    if len(cleaned) < 2:
        return False
    try:
        _, p = stats.levene(*cleaned, center="median")
    except Exception:
        return False
    return bool(p > _ALPHA)


def enough_samples(*groups: Any, min_per_group: int = 20) -> bool:
    """Check whether each group meets the minimum sample size.

    Args:
        *groups: One or more pandas Series (or array-likes), one per group.
        min_per_group: Minimum required non-null observations per group.

    Returns:
        True if every group has at least min_per_group non-null observations.
    """
    if not groups:
        return False
    return all(_clean(g).size >= min_per_group for g in groups)


def expected_counts_ok(table: Any, min_expected: float = 5.0) -> bool:
    """Check whether expected cell counts are sufficient for chi-square.

    If any expected count falls below min_expected, Fisher's exact test should
    be used instead of chi-square.

    Args:
        table: A contingency table (pandas DataFrame or 2-D array) of observed counts.
        min_expected: Minimum acceptable expected count per cell.

    Returns:
        True if all expected counts are >= min_expected.
    """
    observed = np.asarray(table, dtype="float64")
    if observed.ndim != 2 or observed.size == 0:
        return False
    try:
        _, _, _, expected = stats.chi2_contingency(observed)
    except Exception:
        return False
    return bool(np.all(expected >= min_expected))
