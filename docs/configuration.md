# Configuration

## Environment variables

| Variable | Default | What it controls |
|---|---|---|
| `TABULAR_BASE` | process cwd | Where sessions are persisted, **and** the only directory the engine may read data files from |
| `MEELU_ANALYTICS_HOST` | `127.0.0.1` | Bind address for the HTTP transport |
| `MEELU_ANALYTICS_PORT` | `8321` | Bind port for the HTTP transport |

## Transports

By default the server speaks MCP over streamable HTTP at
`http://$MEELU_ANALYTICS_HOST:$MEELU_ANALYTICS_PORT/mcp`. Passing `--stdio` runs
it over stdio instead, for clients that prefer to own the process lifecycle; the
host and port variables are then ignored.

There is no authentication on the HTTP transport. `run_sql` and the ingest tools
are fully available to anything that can open the port, so binding to `0.0.0.0`
is a deliberate decision about who is on your network, not a convenience flag.

## The data directory boundary

`TABULAR_BASE` resolves to the engine's **data root**, and the engine will not
read a file outside it. This is enforced twice:

1. **In DuckDB.** The database is confined to the root at startup and the key is
   thrown away, so no SQL — including a hand-written `read_csv` in `run_sql` —
   can escape it.
2. **In front of it.** `create_session` and `add_table` resolve every path first
   and return a readable `outside_data_dir` error naming the root, because a raw
   DuckDB permission error tells a model nothing about where it *is* allowed to
   look.

The practical consequence: files reach the engine by being downloaded or copied
into `TABULAR_BASE` in the first place. Relative paths, `~` and symlinks are all
resolved before the check, so none of them widen the boundary.

## Storage layout

Sessions live under the base:

```
$TABULAR_BASE/
  .tableint/
    sessions/
      <session_key>/        # DuckDB workspace, metadata, saved models
  your-data.csv             # ingestable files sit anywhere under the base
```

A session survives the process. `list_sessions` enumerates what is on disk, and
any `session_key` is reopened lazily on the first tool call that uses it — a
restarted server picks up exactly where it left off, trained models included.

## Dependencies

**Core** (always installed): ibis-framework[duckdb], pandas, numpy, pydantic,
scipy, scikit-learn, statsmodels, SHAP, and the MCP 1.x SDK.

**Optional extras** — each gates specific tools, imported lazily so a missing
extra costs nothing until you call the tool that needs it:

| Extra | Enables | Install |
|---|---|---|
| `insights` | `market_basket` (mlxtend), `causal_effect` (DoWhy), `detect_changepoints` (ruptures) | `uv sync --extra insights` |
| `dev` | pytest | `uv sync --extra dev` |
| — | `reduce_dimensions(method="umap")` | `uv add umap-learn` |
| — | `train_classifier` / `train_regressor` with `backend="tabicl"` (GPU recommended) | `uv add tabicl` |

Calling a tool whose extra is missing returns an error naming the dependency
rather than crashing the server.
