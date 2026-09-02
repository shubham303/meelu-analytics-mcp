# Feature engineering

Deterministic column builders. Every tool here writes a new column back into the
session's table, and every new column is immediately usable by later models,
tests, and SQL.

Two properties they all share:

- **Out-of-domain values become missing, not errors.** Dividing by zero, taking
  the log of a negative — the row becomes null rather than the call failing. One
  bad row does not cost you the column.
- **The `name` argument is optional.** Omit it and the tool derives a descriptive
  name from the inputs and the operation.

## `combine_columns(session_key, table, col_a, col_b, op, name=None)`

Arithmetic between two numeric columns. `op` is one of `add`, `subtract`,
`multiply`, `divide`, `ratio`. Zero denominators become missing.

This is the primitive behind most domain features. `density = mass / volume`,
`margin = revenue - cost`, `rate = events / exposure` — you supply the columns and
the operation; the arithmetic is generic.

## `transform_column(session_key, table, column, func, name=None)`

Apply a math transform to one numeric column. `func` is one of `log`, `log1p`,
`sqrt`, `square`, `reciprocal`, `abs`, `zscore`. Values outside a transform's
domain — the log of a non-positive, the square root of a negative — become
missing.

Use `log` or `sqrt` to tame right-skew (`profile` reports skewness, which is the
signal), `log1p` when the column contains legitimate zeros, and `zscore` to put
columns on a comparable scale.

## `bin_column(session_key, table, column, n_bins=4, strategy="quantile", name=None)`

Discretize a numeric column into ordinal bins, stored as 0-based integer indices.

| Strategy | Behaviour |
|---|---|
| `quantile` (default) | Equal-frequency — every bin holds roughly the same number of rows |
| `uniform` | Equal-width — every bin spans the same range of values |

They answer different questions. Quantile binning is the right default for skewed
data, where equal-width bins would leave most rows in one bucket; equal-width
bins are what you want when the value ranges themselves are meaningful.

## `expand_datetime(session_key, table, column, parts=None)`

Expand a datetime column into calendar components, each written as
`<column>_<part>`.

Default parts: `year`, `month`, `dayofweek`, `is_weekend`. Available:
`year`, `quarter`, `month`, `week`, `day`, `dayofweek`, `dayofyear`, `hour`,
`is_weekend`, `is_month_start`, `is_month_end`.

A raw timestamp is nearly useless to a model — every value is distinct. The
calendar components are where the seasonality actually lives.

## `group_aggregate(session_key, table, group_by, value, agg="mean", name=None, add_deviation=False)`

Aggregate `value` within each `group_by` category and broadcast the statistic
back to every row: each order gets its customer's mean spend, each transaction
its region's median.

`agg` is one of `mean`, `sum`, `min`, `max`, `std`, `median`, `count`.

This is one of the strongest feature types available on a single table, because
it smuggles relational context into a flat row. With `add_deviation=True` it also
writes `<value>_dev_from_<group_by>` = value − group mean, which is often the
more useful of the two: "this order is £40 above what this customer usually
spends" beats either raw number.

## `row_aggregate(session_key, table, columns, agg="sum", name=None)`

Aggregate several numeric columns *across* each row into one feature. `agg` is
one of `mean`, `sum`, `min`, `max`, `std`, `median`, `count` — where `count` is
the number of non-null inputs.

The generic form of a total: total atom count from per-element columns, total
spend from per-channel columns. `count` doubles as a per-row completeness
measure.

## `normalize_fractions(session_key, table, columns, suffix="_frac")`

Turn a set of count or amount columns into per-row fractions of their total. Each
`<col>` becomes `<col><suffix>` = col / (row sum across the set), so the new
columns sum to 1 for every row.

Composition, separated from magnitude — usually what you actually wanted when
comparing rows of very different sizes.

## `compute_feature(session_key, table, name, expression)`

The escape hatch: one feature column from a custom DuckDB scalar expression,
evaluated per row **inside the database**. Nothing is streamed to the app, so it
works on tables far larger than memory.

```
"mass / NULLIF(volume, 0)"
"CASE WHEN age >= 18 THEN 'adult' ELSE 'minor' END"
"avg(spend) OVER (PARTITION BY customer_id)"
"regexp_extract(email, '@(.*)$', 1)"
```

Strictly feature generation. It must be a single scalar expression, and it must
reference existing columns. Statement chaining, subqueries, DDL/DML, and
file/catalog functions (`read_csv`, `attach`, `install`, …) are rejected — this
is a column builder, not a second SQL surface.
