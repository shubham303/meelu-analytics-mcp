"""Period / group comparison — "what changed, and is the change real?".

``compare_periods`` splits a numeric column into a *before* and *after* window at a
cut date and quantifies the shift: means, delta, % change, a Mann-Whitney test (no
normality assumption), a Kolmogorov-Smirnov distribution test, and Cohen's d effect
size. This is the backbone of any recurring "this month vs last" engagement.

Library: **scipy.stats** (already a core dependency) does every test; we only pick
the split and package the numbers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ....shared import honesty
from ....shared.results import Result


def compare_periods(
    store: Any,
    time_column: str,
    value_column: str,
    split: str | None = None,
) -> Result:
    """Compare ``value_column`` before vs after a cut point on the time axis.

    Args:
        store: The Store/Table instance.
        time_column: Datetime column defining the ordering.
        value_column: Numeric column to compare across the two windows.
        split: ISO date string to split on. Defaults to the median timestamp
            (roughly equal-sized windows).

    Returns:
        Result with before/after means, absolute and % change, Mann-Whitney and
        KS p-values, and Cohen's d.
    """
    frame = store.get_frame()[[time_column, value_column]].copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna().sort_values(time_column)
    if frame.empty:
        raise ValueError("No complete rows across the time and value columns.")

    cut = pd.to_datetime(split) if split else frame[time_column].median()
    before = frame.loc[frame[time_column] < cut, value_column].to_numpy(dtype=float)
    after = frame.loc[frame[time_column] >= cut, value_column].to_numpy(dtype=float)
    if before.size < 2 or after.size < 2:
        raise ValueError(
            f"Each side of the split needs >=2 points; got before={before.size}, after={after.size}."
        )

    mean_before, mean_after = float(before.mean()), float(after.mean())
    delta = mean_after - mean_before
    pct_change = (delta / mean_before * 100.0) if mean_before else None

    mw_p = float(stats.mannwhitneyu(before, after, alternative="two-sided").pvalue)
    ks_p = float(stats.ks_2samp(before, after).pvalue)

    # Cohen's d with pooled SD (guard against zero variance).
    pooled_sd = np.sqrt(((before.std(ddof=1) ** 2) + (after.std(ddof=1) ** 2)) / 2)
    cohens_d = float(delta / pooled_sd) if pooled_sd else 0.0

    direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
    sig = "significant" if mw_p < 0.05 else "not significant"
    pct_txt = f"{pct_change:+.1f}%" if pct_change is not None else "n/a"
    summary = f"{value_column} {direction} {pct_txt} across the split ({sig}, p={mw_p:.3g})"

    # Honesty seam: confidence from the smaller window, plus mandatory caveats.
    trust = honesty.from_sample_size(
        min(int(before.size), int(after.size)), low=30, moderate=200, label="points per window"
    )
    if sig == "not significant":
        trust = honesty.with_caveats(
            trust,
            f"The change is not statistically significant (Mann-Whitney p={mw_p:.3g}) — "
            "it may well be noise.",
        )
    trust = honesty.with_caveats(
        trust,
        "This is a before/after comparison, not a causal test: something other than time "
        "(seasonality, a promo, a mix shift) could explain the change.",
    )

    return Result(
        method="two_window_comparison",
        summary=summary,
        values={
            "mean_before": mean_before,
            "mean_after": mean_after,
            "delta": delta,
            "pct_change": pct_change,
            "mannwhitney_p": mw_p,
            "ks_p": ks_p,
            "cohens_d": cohens_d,
        },
        metadata={
            "time_column": time_column,
            "value_column": value_column,
            "split": cut.isoformat(),
            "n_before": int(before.size),
            "n_after": int(after.size),
        },
        trust=trust,
    )
