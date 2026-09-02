# Descriptive & exploratory tools

The first things to run on a new table, and the fastest way to find what is worth
investigating.

## `profile(session_key, table)`

A per-column summary of the whole table:

- **type** — as classified or pinned (see [Column typing](column-typing.md))
- **missingness** — null count and rate
- **cardinality** — distinct values
- **distribution** — min, max, mean, median, standard deviation, skewness for
  numerics
- **top values** — the most frequent values for categoricals

This is the natural second call after `create_session`. It tells you what the
table actually contains — which columns are usable, which are mostly missing,
which "numeric" column is really an identifier — and therefore what is worth
asking next. Skewness in particular is a hint to reach for
[`transform_column`](feature-engineering.md#transform_columnsession_key-table-column-func-namenone).

## `detect_outliers(session_key, table, column)`

Flag outliers in one numeric column using two methods in **union**:

- **IQR fences** — outside 1.5 × IQR from the quartiles
- **z-score** — |z| > 3

A row flagged by either is flagged, and the flags are written back per row with
attribution for *which* method caught it. The two disagree in informative ways:
z-score assumes roughly normal data and is dragged around by the very outliers it
is looking for, while the IQR fences are robust but indifferent to the
distribution's shape. Seeing which method fired tells you something about the
column.

Flagging is not removal. The column stays as it is; you get a marker to filter
on, investigate, or ignore.

## `analyze_association(session_key, table, col_a, col_b)`

Are these two columns related? The engine picks the correct test from the dtype
pair and the assumption checks, runs it, and reports the method it chose, the
statistic, an effect size, and the assumptions that drove the routing.

The routing table, the assumption checks, and the edge cases have their own page:
**[Association test selection](../association-tests.md)**.

Two things to remember when reading the result:

- **Effect size, not just the p-value.** With enough rows, a trivial difference
  is significant. The magnitude is what tells you whether it matters.
- **Fewer than 10 usable rows is a decline**, and a constant column returns a
  definitive "no association" rather than a fabricated statistic.

## `association_matrix(session_key, table)`

Run the same routing over every testable pair of columns and return a matrix of
effect sizes — each cell routed through `analyze_association`'s logic, so each is
the *appropriate* measure for that pair rather than a correlation forced onto
everything.

This is a scan, not a conclusion. Testing every pair of columns means running a
lot of tests, and some cells will look striking by chance alone. Use it to
generate candidates, then follow up on the interesting ones with
`analyze_association` for the full statistic, assumptions, and trust block.
