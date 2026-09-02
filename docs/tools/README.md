# Tool reference

All 45 MCP tools. Every one takes `session_key` as its first argument, and every
analytic takes a `table` as its second — see the
[session model](../session-model.md).

## By category

| Category | Tools |
|---|---|
| [Session & workspace](session-and-workspace.md) | `create_session`, `list_sessions`, `session_info`, `add_table`, `relationships`, `join`, `run_sql`, `create_table`, `insert_into`, `count_rows`, `count_non_null` |
| [Column typing](column-typing.md) | `list_categorical_columns`, `set_column_type`, `unset_column_type`, `classify_as_nominal` |
| [Descriptive & exploratory](descriptive.md) | `profile`, `detect_outliers`, `analyze_association`, `association_matrix` |
| [Feature engineering](feature-engineering.md) | `combine_columns`, `transform_column`, `bin_column`, `expand_datetime`, `group_aggregate`, `row_aggregate`, `normalize_fractions`, `compute_feature` |
| [Clustering & dim. reduction](clustering-and-dimreduction.md) | `cluster`, `profile_clusters`, `reduce_dimensions` |
| [Supervised ML](supervised-ml.md) | `train_classifier`, `train_regressor`, `evaluate`, `feature_importance`, `add_predictions`, `explain_prediction` |
| [Time series](time-series.md) | `decompose`, `forecast`, `detect_changepoints`, `compare_periods` |
| [Drivers & causal](drivers-and-causal.md) | `explain_metric`, `causal_effect` |
| [Customer analytics](customer-analytics.md) | `market_basket`, `rfm`, `retention_cohorts` |

## A typical order of operations

1. **`create_session`** — ingest, get a key, see detected relationships.
2. **`join`** if the question spans tables. Every analytic runs on one table.
3. **`profile`** — learn the shape of the data before asking anything of it.
4. **Type the columns** — `list_categorical_columns`, then `set_column_type`.
   Statistical routing depends on this.
5. **Explore** — `association_matrix` to find what is worth investigating, then
   `analyze_association` on the interesting pairs.
6. **Engineer** — build the columns the question actually needs.
7. **Model / forecast / explain** — and read the `trust` block on the way out.

## Tools that need optional extras

`market_basket`, `causal_effect`, and `detect_changepoints` need the `insights`
extra; `reduce_dimensions(method="umap")` needs `umap-learn`; the `tabicl`
training backend needs `tabicl`. See
[Configuration](../configuration.md#dependencies).

## Reading results

Analytic tools return `method`, `summary`, `values`, `metadata`, and `trust`.
Check `trust.declined` before using `values`, and pass `trust.caveats` through to
the user. See the [Honesty model](../honesty-model.md).
