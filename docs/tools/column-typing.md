# Column typing

Statistical routing depends on knowing what a column *is*. A number is not
automatically continuous, and a string is not automatically a category. So typing
is explicit, and it is worth doing before you run any test.

## Why it matters

[Association test selection](../association-tests.md) routes on the dtype pair.
Get the type wrong and you get the wrong test — and a wrong test produces a
plausible-looking number, not an error.

Two failure modes are common enough to name:

- **A numeric identifier treated as continuous.** Customer IDs, zip codes, order
  numbers. Correlating one with revenue is meaningless, but it will happily
  return an r.
- **An ordinal treated as nominal.** Ratings, tiers, sizes. Treating them as
  unordered throws away the ordering, which is usually the interesting part.

## Inference on ingest

The engine classifies columns automatically when a table is loaded — continuous,
categorical, datetime, or identifier. Two heuristics are worth knowing:

- A string column whose distinct-value ratio exceeds **0.9** is treated as an
  identifier (UUIDs, emails, primary keys), not a category.
- A numeric column with very few distinct values is treated as categorical rather
  than continuous.

Inference is good enough to explore with, and not good enough to test with. It
lands ambiguous columns on an unrefined **"categorical"** label, which is a
deliberate "you decide" rather than a guess between nominal and ordinal.

## `list_categorical_columns(session_key, table)`

List the columns still sitting at the unrefined `categorical` label — the ones
the engine declined to guess about.

This is the natural call after `profile`, and the work list for the next tool.

## `set_column_type(session_key, table, column, type)`

Pin a column's type. The settable categorical types are:

| Type | Use for |
|---|---|
| `categorical_nominal` | Unordered categories — country, channel, product line |
| `categorical_ordinal` | Ordered categories — rating, tier, size, satisfaction |

Columns are also classified as continuous, datetime, or identifier by inference;
if one of those is wrong, setting the correct categorical type here is how you
override the routing.

## `unset_column_type(session_key, table, column)`

Clear a pinned type and return the column to inference.

## `classify_as_nominal(session_key, table)`

Bulk-classify every remaining unrefined categorical column as nominal.

A convenience for exploratory work, and a real trade-off: it is correct for most
columns and wrong for every ordinal one. Use it to get moving on a wide table,
then fix the ordinals individually with `set_column_type` before you report
anything.
