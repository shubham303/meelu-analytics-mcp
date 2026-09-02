# Architecture

## Layout

```
src/tabint/
  app/                     # composition root — entry points only
    mcp_server.py          #   imports tools, runs FastMCP over HTTP or stdio
  shared/                  # the contracts every feature depends on
    server.py              #   the single FastMCP instance + session registry
    results.py             #   the Result return contract
    honesty.py             #   Trust, trust assessors, decline()
    serialize.py           #   Result → JSON for the wire
    identity.py            #   session/table identity
  analysis/
    tools.py               # all 45 @mcp.tool() definitions — the MCP surface
    session.py             # the Session facade tools call
    db/
      ducktable.py         #   the DuckDB/ibis table handle
      persistence.py       #   session + model storage on disk
    service/
      workspace.py         #   the confined DuckDB workspace, data_root
      relationships.py     #   foreign-key detection
      validation/          #   dtype classification, assumption checks
      algorithms/          #   one module per analytic family
      jobs/                #   job registry and runner
```

## The dependency rule

The arrow points one way:

```
app  →  analysis  →  shared
```

`app` imports `analysis`; features import `shared`; **nothing imports `app`**
except the entry points themselves.

This is what lets each feature's `tools.py` decorate its own functions with
`@mcp.tool()` without importing the composition root. The single FastMCP
instance lives in `shared/server.py`, so registration never creates a cycle, and
`app/mcp_server.py` stays trivial: import the tools module so the decorators run,
then start the server.

The session registry lives in `shared/server.py` for the same reason — both
`analysis/tools.py` and any future transport need to reach it.

## Layers within `analysis`

- **`tools.py`** is the MCP boundary and nothing more. It resolves the session,
  validates and marshals arguments, calls the facade, serializes the result. It
  contains no statistics.
- **`session.py`** is the facade: sessions, tables, and the operations available
  on them. It is the API a Python caller would use directly.
- **`db/`** owns storage — the ibis/DuckDB table handle and on-disk persistence.
- **`service/algorithms/`** is where the actual analytics live, one module per
  family (descriptive, association, clustering, supervised, timeseries, causal,
  basket, cohort, …). These import from `validation` and return `Result`.
- **`service/validation/`** holds dtype classification and the assumption checks
  that drive method routing.

The practical test: an algorithm module should be readable without knowing MCP
exists.

## Two invariants

Everything else is style. These two are the product.

### 1. Method selection is deterministic and recorded

An analytic derives its method from dtypes and assumption checks, never from a
caller's request or a model's guess, and writes both the chosen method and the
checks that produced it into the result. See
[Association test selection](association-tests.md).

### 2. Every result is honest about itself

Every `Result` carries a `Trust`. The serializer emits a `trust` block
unconditionally, defaulting to `unassessed`, so no tool can ship a
confident-looking number by omission. When the data cannot support the question,
the analytic calls `decline(reason)` instead of returning a value. See the
[Honesty model](honesty-model.md).

## Adding an analytic

1. Write the function in the right `service/algorithms/` module. Take a frame or
   a table handle; return a `Result`.
2. Route deterministically. If there is a choice of method, derive it from the
   data and record the reasoning in `metadata`.
3. Assess trust with the helpers in `shared/honesty.py` —
   `from_sample_size`, `with_caveats`, `combine` — and `decline` when the data is
   insufficient. Do not leave it `unassessed` if you can do better.
4. Expose it on the facade in `session.py`.
5. Add the `@mcp.tool()` wrapper in `analysis/tools.py`. The docstring *is* the
   agent-facing documentation — write it for a model that has never seen your
   code.
6. Add a test, and a row in the relevant [tool reference](tools/README.md) page.

## In-database by default

Computation belongs in DuckDB wherever it can be. `compute_feature` evaluates its
SQL expression per row inside the database; feature engineering, aggregation, and
filtering never stream a table through the app. Only the algorithms that
genuinely need a materialized frame — scikit-learn models, SHAP, statsmodels —
pull one.

## The data boundary

The workspace is confined to `TABULAR_BASE` at startup and the key is discarded,
so no SQL can read outside it. `tools.py` checks paths in front of that and
returns a readable error, because the enforcement layer's own message tells a
model nothing useful. See
[the data directory boundary](configuration.md#the-data-directory-boundary).
