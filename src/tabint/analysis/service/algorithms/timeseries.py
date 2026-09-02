"""Time series family — OPTIONAL.

Applicable when the table has a time axis (a datetime column that orders the
rows). decompose splits a series into trend / seasonality / residual; forecast
projects it forward. Library: statsmodels (seasonal_decompose, ARIMA). Prophet
is a possible future lazy path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

from ....shared import honesty
from ....shared.results import Result


def decompose(store: Any, time_column: str, value_column: str) -> Result:
    """Decompose a time series into trend, seasonality, and residual.

    Args:
        store: The Store instance holding the table.
        time_column: Name of the datetime column (the time axis).
        value_column: Name of the numeric column to decompose.
        period is inferred from the series length (falls back to 2).

    Returns:
        Result with trend, seasonality, and residual arrays and the period used.
    """
    series = _ordered_series(store, time_column, value_column)
    if series.size < 4:
        raise ValueError(
            f"decompose needs at least 4 valid observations; {value_column} has {series.size}."
        )
    # seasonal_decompose requires >= 2*period points; never let the period exceed
    # what the (possibly short) series can support.
    period = min(_infer_period(series), series.size // 2)

    # Honesty seam: decomposition needs at least two full seasonal cycles to
    # separate signal from noise; refuse rather than emit a meaningless split.
    if series.size < 2 * period:
        trust = honesty.decline(
            f"Series too short to decompose: {series.size} points is under two full "
            f"cycles at period {period}.",
            caveats=[
                "Decomposition assumes a stable, repeating seasonal pattern; it needs "
                "at least two full periods to estimate seasonality.",
            ],
            basis=[f"n={series.size}", f"period={period}"],
        )
        return Result(
            method="seasonal_decompose_declined",
            summary=f"Declined: series too short to decompose ({series.size} points, period {period}).",
            values={},
            metadata={
                "time_column": time_column,
                "value_column": value_column,
                "period": period,
                "n_points": int(series.size),
            },
            trust=trust,
        )

    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")

    n_cycles = series.size / period
    trust = honesty.from_sample_size(int(series.size), low=24, moderate=60, label="points")
    trust = honesty.with_caveats(
        trust,
        "Decomposition assumes a stable, repeating seasonal pattern; it won't separate "
        "overlapping or changing seasonality well.",
    )
    if n_cycles < 3:
        trust = honesty.with_caveats(
            trust,
            f"Only ~{n_cycles:.1f} seasonal cycles of history — the seasonal estimate is "
            "thin and may absorb one-off events.",
        )

    return Result(
        method="seasonal_decompose",
        summary=f"Decomposed {value_column} (additive, period={period})",
        values={
            "trend": _clean_list(result.trend),
            "seasonal": _clean_list(result.seasonal),
            "residual": _clean_list(result.resid),
        },
        metadata={
            "time_column": time_column,
            "value_column": value_column,
            "period": period,
            "model": "additive",
            "n_points": int(series.size),
        },
        trust=trust,
    )


def forecast(
    store: Any,
    time_column: str,
    value_column: str,
    horizon: int = 10,
) -> Result:
    """Forecast future values with an ARIMA model.

    Args:
        store: The Store instance holding the table.
        time_column: Name of the datetime column.
        value_column: Name of the numeric column to forecast.
        horizon: Number of future periods to forecast.

    Returns:
        Result with point forecasts, 95% confidence intervals, and the ARIMA order.
    """
    series = _ordered_series(store, time_column, value_column)
    n = int(series.size)

    # Honesty seam: too little history to forecast credibly. Refuse rather than
    # extrapolate confidently from a handful of points.
    period = _infer_period(series)
    _FORECAST_CAVEATS = [
        "Forecast uncertainty grows the further out you project.",
        "Assumes the past pattern continues — a regime change (new pricing, a shock) breaks it.",
    ]
    if n < 12 or n < 2 * period:
        trust = honesty.decline(
            f"Only {n} historical points — too short to forecast credibly "
            f"(need at least ~12, and two full cycles at period {period}).",
            caveats=_FORECAST_CAVEATS,
            basis=[f"n={n}", f"period={period}"],
        )
        return Result(
            method="arima_declined",
            summary=f"Declined: history too short to forecast ({n} points).",
            values={},
            metadata={
                "time_column": time_column,
                "value_column": value_column,
                "horizon": horizon,
                "n_points": n,
            },
            trust=trust,
        )

    order = (1, 1, 1)
    model = ARIMA(series.to_numpy(), order=order).fit()
    fc = model.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    # Baseline confidence from history length, then scale DOWN as the horizon
    # grows relative to the history we have — a long horizon on short history is
    # never "high".
    trust = honesty.from_sample_size(n, low=24, moderate=60, label="points")
    ratio = horizon / n
    if ratio > 0.5 and trust.level == honesty.TrustLevel.HIGH:
        trust = honesty.Trust(
            level=honesty.TrustLevel.MODERATE,
            caveats=list(trust.caveats),
            basis=list(trust.basis),
        )
    if ratio > 0.5:
        trust = honesty.with_caveats(
            trust,
            f"The {horizon}-step horizon is large relative to {n} points of history — "
            "later points are extrapolation, treat them as directional at best.",
        )
    trust = honesty.with_caveats(trust, *_FORECAST_CAVEATS)

    return Result(
        method="arima",
        summary=f"{horizon}-step forecast of {value_column} (ARIMA{order})",
        values={
            "forecast": [float(v) for v in mean],
            "lower": [float(v) for v in ci[:, 0]],
            "upper": [float(v) for v in ci[:, 1]],
        },
        metadata={
            "time_column": time_column,
            "value_column": value_column,
            "order": list(order),
            "horizon": horizon,
        },
        trust=trust,
    )


def detect_changepoints(
    store: Any,
    time_column: str,
    value_column: str,
    penalty: float = 10.0,
) -> Result:
    """Detect points in time where the series' behaviour shifts ("sales broke on X").

    Library: **ruptures** (PELT with an RBF cost), imported lazily via the optional
    ``insights`` extra. We return the change *times* plus each segment's mean so a
    finding can read "mean dropped from A to B after <date>".

    Args:
        store: The Store/Table instance.
        time_column: Datetime column (the time axis).
        value_column: Numeric column to scan for shifts.
        penalty: PELT penalty — higher = fewer, more confident changepoints.

    Returns:
        Result with ``changepoints`` (times) and ``segments`` (start/end/mean).
    """
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "detect_changepoints needs ruptures. Install it with:  pip install 'tabint[insights]'."
        ) from exc

    frame = store.get_frame()[[time_column, value_column]].copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.dropna().sort_values(time_column).reset_index(drop=True)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame[values.notna()].reset_index(drop=True)
    signal = frame[value_column].astype(float).to_numpy()
    if signal.size < 4:
        raise ValueError(
            f"detect_changepoints needs at least 4 valid observations; got {signal.size}."
        )

    algo = rpt.Pelt(model="rbf").fit(signal)
    bkps = algo.predict(pen=penalty)  # indices; last is len(signal)

    times = frame[time_column]
    change_times = [times.iloc[i].isoformat() for i in bkps[:-1]]
    bounds = [0, *bkps]
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = signal[start:end]
        segments.append({
            "start": times.iloc[start].isoformat(),
            "end": times.iloc[end - 1].isoformat(),
            "mean": float(seg.mean()),
            "n": int(seg.size),
        })

    summary = (
        f"{len(change_times)} changepoint(s) in {value_column}"
        + (f"; first at {change_times[0]}" if change_times else "")
    )

    # Honesty seam: more points → shifts are more reliably located. A lone
    # changepoint on a short/noisy series is weak evidence.
    trust = honesty.from_sample_size(int(signal.size), low=30, moderate=100, label="points")
    trust = honesty.with_caveats(
        trust,
        "Detects WHERE the series shifts statistically, not WHY — pair with domain knowledge.",
    )
    if signal.size < 30 and len(change_times) <= 1:
        trust = honesty.with_caveats(
            trust,
            "A single changepoint on a short series can reflect noise as easily as a real "
            "regime change — corroborate before acting on it.",
        )

    return Result(
        method="ruptures_pelt_rbf",
        summary=summary,
        values={"changepoints": change_times, "segments": segments},
        metadata={
            "time_column": time_column,
            "value_column": value_column,
            "penalty": penalty,
            "n_points": int(signal.size),
        },
        trust=trust,
    )


def _ordered_series(store: Any, time_column: str, value_column: str) -> pd.Series:
    """Return the value column as a numeric Series ordered by the time column."""
    frame = store.get_frame()[[time_column, value_column]].copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.dropna().sort_values(time_column)
    series = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    series.index = frame.loc[series.index, time_column]
    return series.reset_index(drop=True)


def _infer_period(series: pd.Series) -> int:
    """A conservative seasonal period: enough data for two full cycles, else 2."""
    n = series.size
    for candidate in (12, 7, 4):
        if n >= 2 * candidate:
            return candidate
    return max(2, n // 2)


def _clean_list(arr: Any) -> list[float]:
    return [None if pd.isna(v) else float(v) for v in np.asarray(arr)]
