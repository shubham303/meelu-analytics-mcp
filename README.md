# Meelu analytics MCP server

Deterministic single-table data analysis as a standalone MCP server over
streamable HTTP. Load CSVs into a DuckDB-backed session, then run profiling,
statistical tests, machine learning, forecasting, and causal inference through
45 MCP tools. Test and algorithm selection is deterministic — the engine picks
the right method from the data's shape and records why. Ported from
[TableIntelligence](https://github.com/shubham303/TableIntelligence) (same
author).

## Run the server

```bash
TABULAR_BASE="$HOME/.meelu-open/analytics" uv run --project <this dir> meelu-analytics-mcp
```

`uv` resolves and installs the dependencies into `.venv` on the first run —
that is the whole install story. The server listens on
`http://127.0.0.1:8321/mcp`; override with `MEELU_ANALYTICS_HOST` /
`MEELU_ANALYTICS_PORT`. Sessions (DuckDB files, saved state) live under
`TABULAR_BASE` (default: the process cwd), so point it somewhere stable.
Pass `--stdio` to run over stdio instead of HTTP.

## Connect an agent

Claude Code:

```bash
claude mcp add --transport http meelu-analytics http://127.0.0.1:8321/mcp
```

Any other MCP client: point it at `http://127.0.0.1:8321/mcp` with the
streamable HTTP transport. The server binds to localhost by default; set
`MEELU_ANALYTICS_HOST=0.0.0.0` only if you understand who can reach the port —
there is no authentication.

## The session model

1. `create_session(paths)` ingests CSV files and returns a `session_key`. It
   also detects foreign-key relationships between the uploaded tables.
2. Every later tool call carries that key — the data is never re-sent.
3. Every analytic runs on one table. For related tables, `join` materializes a
   combined table first.
4. Results come back with the chosen method, a one-line summary, the values,
   and a `trust` block (see [Honesty model](#honesty-model)).

## Tool catalog

### Session and workspace

| Tool | What it does |
|---|---|
| `create_session` | Ingest CSV files, detect foreign-key relationships, return a session key |
| `list_sessions` / `session_info` | Enumerate sessions; summarize one session's tables |
| `add_table` | Add another CSV to an existing session |
| `relationships` | Show detected foreign-key links between tables |
| `join` | Materialize a combined table from related tables (left join by default) |
| `run_sql` | Run arbitrary SQL against the session's DuckDB workspace |
| `create_table` / `insert_into` | Create a table from a schema or SQL; append rows from a SQL source |
| `count_rows` / `count_non_null` | Row count; non-null count for one column |

### Column typing

Statistical routing depends on knowing whether a column is continuous,
nominal, or ordinal, so typing is explicit.

| Tool | What it does |
|---|---|
| `list_categorical_columns` | List columns still at the unrefined "categorical" label |
| `set_column_type` / `unset_column_type` | Pin a column to continuous / nominal / ordinal / datetime / identifier, or clear it |
| `classify_as_nominal` | Bulk-classify all unrefined categorical columns as nominal |

### Descriptive and exploratory

| Tool | Algorithm |
|---|---|
| `profile` | Per-column type, missingness, cardinality, min/max/mean/median/std, skewness, top values |
| `detect_outliers` | IQR fences (1.5×IQR) and z-score (\|z\| > 3) in union; flags written back per row with method attribution |
| `analyze_association` | Deterministic test routing by dtype pair — see [Association test selection](#association-test-selection) |
| `association_matrix` | Pairwise effect-size matrix over all testable columns, each cell routed through `analyze_association` |

### Feature engineering

Deterministic column builders; every new column is written back and becomes
usable in later models and queries.

| Tool | Operations |
|---|---|
| `combine_columns` | add, subtract, multiply, divide/ratio (zero denominators become missing) |
| `transform_column` | log, log1p, sqrt, square, reciprocal, abs, z-score (out-of-domain values become missing) |
| `bin_column` | Quantile (equal-frequency) or uniform (equal-width) binning |
| `expand_datetime` | Extract year, quarter, month, week, day, day-of-week, and friends |
| `group_aggregate` | Per-group mean/sum/min/max/std/median/count broadcast back to rows |
| `row_aggregate` | The same aggregates across columns within each row |
| `normalize_fractions` | Turn count columns into fractions of a row total |
| `compute_feature` | Arbitrary SQL expression evaluated per row in-database |

### Clustering and dimensionality reduction

| Tool | Algorithm |
|---|---|
| `cluster` | k-means (scikit-learn) on scaled, one-hot-encoded features; k in [2, 10] chosen by maximum silhouette score when not given; labels written back |
| `profile_clusters` | Per-cluster size, numeric means, dominant categories |
| `reduce_dimensions` | PCA (with explained-variance ratios), t-SNE (perplexity adapted to n), or UMAP (optional dependency); components written back as columns |

### Supervised machine learning

| Tool | Algorithm |
|---|---|
| `train_classifier` / `train_regressor` | scikit-learn Pipeline (impute + one-hot/ordinal encode) with a 75/25 stratified train/test split. Backends: `gbt` (default, HistGradientBoosting) or `tabicl` (TabICL v2 tabular foundation model, optional dependency) |
| `evaluate` | On the held-out split — classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix; regression: MAE, RMSE, R² |
| `feature_importance` | Permutation importance (model-agnostic, 10 repeats, held-out split), aggregated to original columns |
| `add_predictions` | Write model predictions back as a column |
| `explain_prediction` | SHAP local explanation for one row (exact TreeExplainer for the gradient-boosted default), contributions summed back to original columns |

Training refuses under 30 usable rows, or when a class has fewer than 2
examples — a model whose metrics are meaningless is worse than a refusal.

### Time series

| Tool | Algorithm |
|---|---|
| `decompose` | Additive seasonal decomposition (statsmodels `seasonal_decompose`) into trend, seasonality, residual; period inferred from series length |
| `forecast` | ARIMA(1,1,1) (statsmodels) point forecasts with 95% confidence intervals; refuses on under ~12 points or under two seasonal cycles |
| `detect_changepoints` | PELT with an RBF cost (ruptures, optional dependency); returns change times plus per-segment means |
| `compare_periods` | Before/after split at a cut date: mean delta and % change, Mann-Whitney U, Kolmogorov-Smirnov, Cohen's d |

### Key drivers and causal inference

| Tool | Algorithm |
|---|---|
| `explain_metric` | Shallow decision tree (depth ≤ 3, leaves ≥ 5% of rows) over all other columns: ranked feature importances plus human-readable segment rules |
| `causal_effect` | DoWhy (optional dependency): backdoor adjustment via linear regression with all other usable columns as default confounders, then a random-common-cause placebo refutation. A failed refutation withholds the effect estimate entirely; observational estimates never earn more than moderate trust |

### Customer analytics

| Tool | Algorithm |
|---|---|
| `market_basket` | Apriori frequent itemsets + association rules (mlxtend, optional dependency), ranked by lift with support/confidence thresholds |
| `rfm` | Recency/Frequency/Monetary quintile scoring into the canonical segments (Champions, Loyal, Potential Loyalist, At Risk, Hibernating) |
| `retention_cohorts` | Monthly retention matrix: first-purchase cohort × months-since-first, counts and rates |

## Association test selection

`analyze_association` routes to the correct test purely from the dtype pair,
checking normality, equal variance, sample size, and expected cell counts
first. Every choice — and the assumption checks that drove it — is recorded in
the result's metadata.

| Column pair | Assumptions hold | Assumptions fail | Effect size |
|---|---|---|---|
| continuous × continuous | Pearson correlation (both normal) | Spearman rank correlation | r / ρ, plus r² |
| categorical × continuous, 2 groups | Welch's t-test (no equal-variance assumption) | Mann-Whitney U | η² and ω² (parametric) / ε² (rank-based) |
| categorical × continuous, 3+ groups | One-way ANOVA (normal + equal variance) | Kruskal-Wallis | η² and ω² / ε² |
| categorical × categorical | Chi-square (expected counts OK) | Fisher's exact (2×2, small counts) | Cramér's V |

Degenerate cases (a constant column) return a definitive "no association"
rather than a fabricated statistic; fewer than 10 usable rows is a decline.

## Honesty model

Every result carries a `trust` block — a confidence level (high / moderate /
low / none / unassessed) with caveats and the basis behind it — and a
`declined` flag. Confidence scales with sample size and quality signals
(silhouette for clustering, variance explained for PCA, refutation outcome for
causal estimates). When the data cannot support the question — too few rows, a
single-valued treatment, a failed placebo test — the tool declines with a
reason instead of returning a number. Agents must report declines and caveats
rather than substituting a value.

## Optional extras

| Extra | Enables | Install |
|---|---|---|
| `insights` | `market_basket` (mlxtend), `causal_effect` (DoWhy), `detect_changepoints` (ruptures) | `uv sync --extra insights` |
| — | `reduce_dimensions(method="umap")` | `uv add umap-learn` |
| — | `train_*(backend="tabicl")` foundation-model backend (GPU recommended) | `uv add tabicl` |

Core (always installed): DuckDB/ibis, pandas, numpy, scipy, scikit-learn,
statsmodels, SHAP.

Requires Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/).
