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

`TABULAR_BASE` resolves to the engine's **data root**. The boundary it draws is
about *improvised* file access, not about ingest.

**In DuckDB.** The database is confined to the root at startup and the key is
thrown away, so no SQL — including a hand-written `read_csv` in `run_sql`,
`create_table` or `insert_into` — can escape it. This is the boundary that
matters: it stops an arbitrary-file read from hiding inside a plain-looking
`SELECT`.

**Ingest is not subject to it.** `create_session` and `add_table` accept a path
anywhere this process can read. Loading a CSV is a one-way copy into DuckDB —
nothing afterwards refers back to the file — so requiring users to first move a
spreadsheet into a blessed folder bought friction, not safety.

A path outside the root is served by copying the file into a unique directory
under `$TABULAR_BASE/.staging`, reading it from there, and deleting the copy once
the rows are in. The database's own allow-list is never widened, so an ingest
cannot unseal `run_sql` for the rest of the session. The table still takes its
name from the original filename.

A path that does not resolve to a readable file returns a `file_not_found` error.
Relative paths, `~` and symlinks are all resolved first.

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
| `foundation` | *Optional preinstall only.* The TabICL v2 foundation model is normally installed on demand by the `install_foundation_model` tool (see below); this extra puts it in the server's own environment instead. Adds PyTorch | `uv sync --extra foundation` |
| `dev` | pytest | `uv sync --extra dev` |
| — | `reduce_dimensions(method="umap")` | `uv add umap-learn` |

Calling a tool whose extra is missing returns an error naming the dependency
rather than crashing the server.

**The foundation model is never part of the server install.** Until an agent
calls `install_foundation_model`, `train_*` uses gradient-boosted trees, records
why in `metadata.backend_selection`, and returns a `hint` saying how to install
it. The tool installs PyTorch (CPU) and TabICL into `<TABULAR_BASE>/.deps/foundation`
— outside the package's environment, so it survives upgrades — and downloads the
weights, in the background. See [Tabular foundation model](tools/foundation-models.md).

| Variable | Default | Effect |
|---|---|---|
| `MEELU_FOUNDATION_AUTO_INSTALL` | unset | `1` starts the background install automatically the first time `auto` would have used the model (the call itself still trains trees) |
| `MEELU_FOUNDATION_DEVICE` | `cpu` | Torch device for TabICL. `cuda` uses a GPU (faster; results may differ in the last digits) |
| `HF_HOME` / `HF_HUB_CACHE` | `~/.cache/huggingface` | Where the ~220 MB of weights are cached |
| `HF_HUB_OFFLINE` | unset | `1` never touches the network; the install step then fails with a reason and `auto` keeps using trees |
