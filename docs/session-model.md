# The session model

Every tool in this server operates inside a **session**. Understanding the
session model is most of understanding the API.

## Why sessions exist

Analysis is iterative: profile, type a column, engineer a feature, train a model,
explain a prediction. If each of those calls had to carry the data, an agent
would spend its entire context window re-transmitting a CSV. So the data is
uploaded once, into a DuckDB workspace on disk, and every later call carries only
a `session_key`.

That also makes write-back possible. Cluster labels, predictions, engineered
features, and PCA components are stored as real columns in the workspace, so the
next tool — and any `run_sql` query — sees them.

## The lifecycle

### 1. Create

```
create_session(paths=["…/orders.csv", "…/customers.csv"])
→ { "session_key": "…", "tables": ["orders", "customers"], "relationships": {…} }
```

Each CSV becomes a table named after the file. Ingest also runs foreign-key
detection across the uploaded tables, so the relationships come back in the same
response.

Paths may point anywhere this process can read; the rows are copied into the
session's database on load — see
[the data directory boundary](configuration.md#the-data-directory-boundary).

### 2. Work

Pass `session_key` to every subsequent call. Add more data later with
`add_table`, build clean tables with `create_table` / `insert_into`, query
anything with `run_sql`.

### 3. Reopen

Sessions are persisted, not in-memory. `list_sessions` enumerates the keys on
disk, and any key is reopened lazily on first use — including saved models. A
restarted server loses nothing.

## The one-table rule

**Every analytic runs on exactly one table.** There is no analytic that spans
tables implicitly.

This is a deliberate constraint. A statistical test over an implicit join has a
row count that nobody can reason about: a one-to-many join inflates n, and every
p-value computed from it is wrong. Forcing the join to be explicit makes the unit
of analysis visible and the sample size real.

So for related tables:

```
relationships(session_key)                          # what links to what
join(session_key, tables=["orders", "customers"])   # materialize the combination
profile(session_key, table="orders_customers")      # then analyze the result
```

`join` defaults to a left join and writes a genuine new table into the workspace.
It is a first-class table from then on: you can engineer features on it, train
models against it, and join it again.

## Relationship detection

On ingest, the engine looks for foreign-key relationships between tables — column
name and type compatibility, value containment, and cardinality. `relationships`
returns the detected links with the evidence behind each.

Detection is a suggestion, not a fact. Review the links before joining on them;
`join` will use what you tell it.

## Tables you build yourself

Real data is frequently the wrong shape. `run_sql` deliberately cannot create
tables, so the path is explicit:

- `create_table(name, columns=[…])` — an empty table with the schema you want,
  then `insert_into(name, source_sql)` to copy data across, once or repeatedly
  from many messy sources.
- `create_table(name, select_sql="…")` — materialize a query in one shot.

Either way the result is a normal session table, indistinguishable from an
ingested one.

## Result shape

Analytic tools return a uniform structure:

| Field | Meaning |
|---|---|
| `method` | What was actually run — the chosen test or algorithm, not what you asked for |
| `summary` | One line of plain language |
| `values` | The statistics, scores, coefficients |
| `metadata` | Assumption checks and parameters — the record of *why* this method |
| `trust` | Confidence level, caveats, basis, and the `declined` flag |

See the [Honesty model](honesty-model.md) for how to read `trust`.
