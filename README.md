# meelu-analytics-mcp

**Deterministic single-table data analysis as a standalone MCP server.**

Load CSVs into a DuckDB-backed session, then run profiling, statistical tests,
feature engineering, machine learning, forecasting, and causal inference through
45 MCP tools — from Claude Code, or any MCP client.

The point is the *determinism*. You do not ask for a Pearson correlation; you ask
whether two columns are associated, and the engine picks the correct test from the
data's shape — checking normality, variance, and cell counts first — then records
which test it chose and why. Every result also says how much to trust it, and
refuses to answer when the data cannot support the question.

```bash
uv run --project . meelu-analytics-mcp
claude mcp add --transport http meelu-analytics http://127.0.0.1:8321/mcp
```

→ **[Getting started](docs/getting-started.md)** for the full setup.

## Documentation

| Guide | What's in it |
|---|---|
| [Getting started](docs/getting-started.md) | Install, run the server, connect an agent, first analysis |
| [Configuration](docs/configuration.md) | Environment variables, storage layout, optional extras |
| [Session model](docs/session-model.md) | Sessions, tables, the one-table rule, persistence |
| [Honesty model](docs/honesty-model.md) | Trust levels, caveats, declines — and how to read them |
| [Association test selection](docs/association-tests.md) | The deterministic routing table, in detail |
| [Architecture](docs/architecture.md) | Module layout and the dependency rules behind it |
| **[Tool reference](docs/tools/README.md)** | All 45 tools, by category |

### Tool reference by category

- [Session & workspace](docs/tools/session-and-workspace.md) — ingest, join, SQL, table building
- [Column typing](docs/tools/column-typing.md) — the typing that drives statistical routing
- [Descriptive & exploratory](docs/tools/descriptive.md) — profile, outliers, association
- [Feature engineering](docs/tools/feature-engineering.md) — deterministic column builders
- [Clustering & dimensionality reduction](docs/tools/clustering-and-dimreduction.md)
- [Supervised machine learning](docs/tools/supervised-ml.md) — train, evaluate, explain
- [Time series](docs/tools/time-series.md) — decompose, forecast, changepoints
- [Drivers & causal inference](docs/tools/drivers-and-causal.md)
- [Customer analytics](docs/tools/customer-analytics.md) — basket, RFM, cohorts

## What you get

- **45 MCP tools** over one coherent session model — data is uploaded once and
  never re-sent.
- **Deterministic method selection.** Test and algorithm choice follows from
  dtypes and assumption checks, not from an agent's guess.
- **Refusals over fabrications.** Under 30 rows, a single-valued treatment, a
  failed placebo test — the tool declines with a reason instead of returning a
  meaningless number.
- **Everything writes back.** Cluster labels, predictions, engineered features,
  PCA components all become real columns, usable by later tools and SQL.
- **In-database by default.** DuckDB/ibis does the work; nothing streams through
  the app that doesn't have to.

## Roadmap

Phases are ordered so each makes the next more useful. ✅ ships today.

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation — workspace, `Result`, dtype validation, sessions, persistence | ✅ |
| 1 | Descriptive — `profile`, `detect_outliers`, `association_matrix` | ✅ |
| 2 | Association — dtype-driven test routing (the flagship) | ✅ |
| 3 | Clustering — `cluster`, `profile_clusters`, write-back machinery | ✅ |
| 4 | Supervised — train / evaluate / predict, plus the job model | ✅ |
| 5 | Interpretation — `feature_importance`, `explain_prediction` | ✅ |
| 6 | Dimensionality reduction — PCA, t-SNE, UMAP | ✅ |
| 7 | Time series — decompose, forecast, changepoints, period comparison | ✅ |
| 8 | Insights — causal effects, key drivers, basket, RFM, cohorts | ✅ |
| 9 | MCP surface — standalone streamable-HTTP server | ✅ |

### Next

- **Large-data strategies** — sampling, out-of-core execution, approximate
  methods, so the engine degrades gracefully instead of refusing.
- **PyPI distribution** — `uvx meelu-analytics-mcp` with no local checkout.
- **Broader ingest** — Parquet and JSON alongside CSV; connector-fed tables.
- **Richer trust assessment** — replace the remaining `unassessed` results with
  real, method-specific confidence.
- **Multi-table analytics** — reduce the reliance on an explicit `join` step.

### Deliberately out of scope

Publishing, reporting, and external connectors. This engine computes; something
else decides what to do with the answer.

## Contributing

Issues and pull requests are welcome. When adding an analytic, keep the two
invariants: method selection must be deterministic and recorded in the result's
metadata, and every result must carry an honest `trust` block — including a
decline when the data cannot support the question. See
[Architecture](docs/architecture.md) for where things belong.

```bash
uv sync --extra dev --extra insights
uv run pytest
```

## Credits and license

Ported from [TableIntelligence](https://github.com/shubham303/TableIntelligence)
by the same author. Licensed under the [MIT License](LICENSE).
