# Session & workspace tools

Ingest, inspect, combine, and build tables. See
[the session model](../session-model.md) for the concepts.

## `create_session(paths)`

Ingest one or more CSV files and return a `session_key`, the resulting table
names, and the foreign-key relationships detected between them.

Each file becomes a table named after it. Every path must resolve inside the
engine's data root, or the call returns an `outside_data_dir` error naming the
root — see
[the data directory boundary](../configuration.md#the-data-directory-boundary).

## `list_sessions()`

The keys of all sessions persisted under the current base. Sessions outlive the
process, so this includes sessions from previous runs.

## `session_info(session_key)`

Summarize one session: its tables and their detected relationships.

## `add_table(session_key, path)`

Ingest another CSV into an existing session. Relationship detection re-runs, so
the new table's links to the existing ones are picked up.

## `relationships(session_key)`

The detected foreign-key links between the session's tables, with the evidence
behind each — name and type compatibility, value containment, cardinality.

Detection is a suggestion. Review the links before joining on them.

## `join(session_key, tables, name=None, how="left")`

Materialize a combined table from related tables, using the detected
relationships. Defaults to a left join; `name` defaults to a name derived from
the inputs.

The result is a real table in the workspace, not a view — a first-class citizen
you can profile, engineer features on, model against, and join again.

Because every analytic runs on one table, this is how multi-table questions get
asked. Making it explicit keeps the unit of analysis and the row count visible;
an implicit one-to-many join would silently inflate n and invalidate every
p-value computed from it.

## `run_sql(session_key, query, limit=1000)`

Run arbitrary SQL against the session's DuckDB workspace. Returns up to `limit`
rows.

The workspace is confined to the data root, so no query can read outside it. Use
this for filtering, inspection, and shaping questions the fixed tools do not
cover. It **cannot create tables** — that is `create_table`'s job, deliberately,
so that table creation is an explicit, reviewable act.

## `create_table(session_key, name, columns=None, select_sql=None)`

Create a new clean, structured table. Use it when the source data is messy or
badly shaped: define the schema you want, then copy data into it.

Exactly one of two mutually exclusive modes:

- **`columns`** — an empty typed table. Each entry is
  `{"name": …, "type": …}`, e.g.
  `[{"name": "order_id", "type": "BIGINT"}, {"name": "amount", "type": "DECIMAL(10,2)"}]`.
  Types are standard SQL/DuckDB types: `INTEGER`, `BIGINT`, `DOUBLE`,
  `DECIMAL(p,s)`, `VARCHAR`, `DATE`, `TIMESTAMP`, `BOOLEAN`, …
- **`select_sql`** — materialize a query over the existing tables in one shot,
  e.g. `SELECT trim(name) AS name, CAST(qty AS INTEGER) AS qty FROM raw`.

Returns the created table's name, columns, and row count.

## `insert_into(session_key, name, source_sql)`

Copy rows into an existing table — the partner of `create_table`'s `columns`
mode. `source_sql` is a `SELECT` or `VALUES` query whose columns map
**positionally** to the target's:

```sql
SELECT trim(customer), CAST(spend AS DECIMAL(10,2)) FROM raw WHERE spend IS NOT NULL
```
```sql
VALUES ('Acme', 12.50), ('Globex', 9.99)
```

Call it repeatedly to build one clean table from many messy sources. Returns the
rows inserted and the table's new total.

## `count_rows(session_key, table)`

The table's row count.

## `count_non_null(session_key, table, column)`

The non-null count for one column. Worth checking before an analysis: the usable
n, not the table length, is what determines whether a result means anything.
