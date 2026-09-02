# Time series

Every tool here takes a `time_column` and a `value_column`, and every one has a
minimum data requirement it will decline below rather than extrapolate from
nothing.

## `decompose(session_key, table, time_column, value_column)`

Additive seasonal decomposition (statsmodels `seasonal_decompose`) into three
components: **trend**, **seasonality**, and **residual**.

The period is inferred from the series length. This is the right first call on
any time series, because it separates questions that get confused when asked of
the raw line: "are we growing?" is about the trend, "is December always like
this?" is about the seasonal component, and "was last week unusual?" is about the
residual.

The decomposition is additive — components sum to the observed series. For data
whose seasonal swings scale with the level (a peak that is always +20%, not
always +200 units), apply a log
[`transform_column`](feature-engineering.md#transform_columnsession_key-table-column-func-namenone)
first; on the log scale, multiplicative seasonality becomes additive.

## `forecast(session_key, table, time_column, value_column, horizon=10)`

ARIMA(1,1,1) point forecasts with 95% confidence intervals, `horizon` steps
ahead.

The order is fixed rather than searched: one autoregressive term, one difference,
one moving-average term. It is a reasonable general-purpose model for a trending
series and a deliberately modest one. It does not model seasonality, so run
`decompose` first — if the seasonal component is large, treat these forecasts as
a trend line, not a prediction.

**Report the intervals, always.** The point forecast is the least informative
part of the output. A forecast whose 95% interval spans an order of magnitude is
telling you the honest answer is "we don't know", and quoting only the central
value hides that entirely. Intervals also widen with the horizon, which is why
asking for 200 steps produces something closer to art than analysis.

Forecasting **declines** below roughly 12 points, or below two full seasonal
cycles.

## `detect_changepoints(session_key, table, time_column, value_column, penalty=10.0)`

Find the points where a series shifts behaviour — PELT with an RBF cost
(`ruptures`, needs the [`insights` extra](../configuration.md#dependencies)).
Returns the change times plus per-segment means.

`penalty` controls sensitivity: **higher means fewer changepoints**. The default
of 10.0 is a starting point, not a calibrated value — the right penalty depends
on the scale and noisiness of your series, so sweep it and see which structure is
stable across values. Structure that appears only at one penalty is usually
noise.

This answers a question no forecast can: not "what happens next" but "when did
something change". Useful for finding the date an intervention actually took
effect, rather than the date it was announced.

## `compare_periods(session_key, table, time_column, value_column, split=None)`

Split the series at a cut date and compare before vs after:

- **mean delta** and **% change**
- **Mann-Whitney U** — a rank-based test of whether the distributions differ
- **Kolmogorov-Smirnov** — sensitive to any distributional difference, not just
  location
- **Cohen's d** — the standardized effect size

Rank-based and distribution-free tests are used deliberately: period data is
rarely normal, and the two tests answer different questions — Mann-Whitney asks
whether values are typically larger after, KS asks whether the whole shape
changed.

Cohen's d is the number to lead with. A significant p-value on a long series can
accompany a change too small to act on.

**This is not causal inference.** A before/after difference is confounded by
everything else that changed at the same time — seasonality, trend, other
interventions. `compare_periods` measures the difference; it does not attribute
it. See [`causal_effect`](drivers-and-causal.md) for that question, and its own
substantial caveats.
