# Backlog

The work queue for the autonomous `ship-feature` skill (`.claude/skills/ship-feature`).
Items are ordered by priority (value ÷ effort); the agent takes the first `pending`
item that has no open PR labelled `autonomous`. Statuses: `pending` → `in-review`
(PR open) → `done` (merged), or `blocked` with a reason. Add new items with the next
free ID, in priority position, using the same fields.

## B-001 Stop stacked SQL statements in run_sql, create_table and insert_into
- **Status:** pending
- **Type:** bug
- **Priority:** P0
- **Effort:** S
- **Why:** `run_sql` is documented as "read-only SELECT", but I confirmed that `run_sql("SELECT 1; DROP TABLE _tbl_orders")` drops the table. `create_table(select_sql=...)` and `insert_into(source_sql=...)` also run stacked statements. That means an agent can destroy session data or rewrite `_ti_column_types` and `_ti_derived_columns`, which hold the method-routing and leakage metadata the honesty model depends on.
- **Scope / acceptance criteria:**
  - Add one helper, e.g. `workspace._require_single_select(sql)`, that uses `duckdb.extract_statements` or a sqlglot parse. It must accept exactly one statement, and that statement must be SELECT, WITH…SELECT or VALUES.
  - Call it in `Workspace.run_sql`, `create_table(select_sql=…)` and `insert_into`.
  - Refuse any reference to internal objects (`_tbl_*`, `_ti_*`, `_seq_*`) from agent-authored SQL. A simple approach is to check the table names in the parsed statement.
  - Tests: `"SELECT 1; DROP TABLE _tbl_orders"`, `"SELECT 1 AS a); DROP TABLE _tbl_orders; SELECT * FROM (SELECT 1 AS a"` in `create_table`, and `"SELECT …; DELETE FROM _tbl_orders"` in `insert_into` all raise. The table row count is unchanged afterwards.
  - End-to-end check: in an MCP client, call `run_sql` with a stacked DROP. You should get an error, and `count_rows` should still return the original count.
- **Pointers:** `src/tabint/analysis/service/workspace.py:558-560` (run_sql), `:598` (create_from_query), `:613` (insert_into); `src/tabint/analysis/db/ducktable.py:44-56`, `:82-96`; `src/tabint/analysis/tools.py:155-209`. sqlglot (MIT) is already pulled in by ibis.

## B-002 Installer leaves out the `insights` extra, so three tools fail for every installed user
- **Status:** pending
- **Type:** bug
- **Priority:** P0
- **Effort:** S
- **Why:** `scripts/install.py` runs `uv tool install meelu-analytics-mcp` without `[insights]`. As a result, `market_basket`, `causal_effect` and `detect_changepoints` raise ImportError for everyone who used the one-command installer. The README advertises all three. The error hints also say `pip install 'tabint[insights]'`, but that package does not exist.
- **Scope / acceptance criteria:**
  - Choose one of these:
    - install `meelu-analytics-mcp[insights]` (and the git-URL equivalent) in `install_server` and the uvx fallback, or
    - move mlxtend, dowhy and ruptures into core dependencies.
  - Change every hint string to `uv tool install 'meelu-analytics-mcp[insights]'`.
  - The warm-up step should actually import shap and dowhy, e.g. a `--warmup` flag that imports the heavy modules and exits. Today shap is wrapped in `_lazy_import`, so `--stdio` with no stdin probably never compiles the numba kernels the docstring talks about.
  - Verify: a fresh `install.sh --agent claude-code --yes`, then call `detect_changepoints` through MCP. It should return a result, not an ImportError.
- **Pointers:** `scripts/install.py:159-191`, `:140-156`; `src/tabint/analysis/service/algorithms/{basket.py:46,causal.py:100,timeseries.py:217,supervised.py:87}`; `pyproject.toml:342-348`.

## B-003 Run tests in CI on every push and pull request
- **Status:** pending
- **Type:** engineering
- **Priority:** P0
- **Effort:** S
- **Why:** Tests run only inside the tag-triggered publish workflow. Regressions like B-001 and B-002 reach `main`, and the installer pulls straight from `main`.
- **Scope / acceptance criteria:**
  - Add `.github/workflows/ci.yml`, triggered on `push` and `pull_request`.
  - Use `astral-sh/setup-uv` and run `uv sync --extra dev --extra insights`, then `uv run pytest -q`.
  - Matrix: Python 3.10 and 3.13, on ubuntu-latest and macos-latest. Windows is added in B-026.
  - Add a lint job (`ruff check`) and a job that runs `python -c "import tabint.app.mcp_server"` to catch import-time breakage.
  - Add a "lowest-direct" resolution job (`uv sync --resolution lowest-direct`) so the dependency floors are tested too.
  - Make the publish workflow depend on CI passing, or have it reuse CI.
  - Verify: the pull request shows green checks, and a deliberately broken test turns them red.
- **Pointers:** `.github/workflows/publish-mcp.yml`, `pyproject.toml` (add `ruff` to the `dev` extra).

## B-004 Unset `TABULAR_BASE` puts the SQL sandbox at the process's working directory
- **Status:** pending
- **Type:** bug
- **Priority:** P0
- **Effort:** S
- **Why:** When `TABULAR_BASE` is unset, the base defaults to `"."`. A server launched by `uvx`, a hand-written config or an IDE often has `$HOME` or `/` as its working directory. DuckDB's `allowed_directories` then covers the whole home directory, and `run_sql("SELECT * FROM read_csv_auto('~/.ssh/…')")` becomes possible. Sessions are also written to random places.
- **Scope / acceptance criteria:**
  - Default to a dedicated directory, `~/meelu-data` (matching the installer), not the working directory.
  - Refuse to start, with a clear message, if the resolved base is `/`, the home directory itself, or a filesystem root.
  - Do the same in `data_root()` and `shared/server.py`.
  - Log the resolved data root to stderr at startup.
  - Update `docs/configuration.md` (the table currently says "process cwd") and the `server.json` description, which says it is "the only directory the server is permitted to read from". That is no longer true because of staging.
  - Tests: with no env var and the working directory set to a tmp "home", the confine root is `<home>/meelu-data`. A base equal to `Path.home()` raises.
- **Pointers:** `src/tabint/shared/server.py:40`, `src/tabint/analysis/service/workspace.py:428-437`, `src/tabint/analysis/db/persistence.py:41-43`, `server.json`, `docs/configuration.md:7`.

## B-005 Add a unit test suite for every algorithm family
- **Status:** pending
- **Type:** engineering
- **Priority:** P0
- **Effort:** M
- **Why:** The only test file covers file containment. Nothing checks the product's core promise that method routing is deterministic and that declines happen when the data is too thin.
- **Scope / acceptance criteria:**
  - Add a `tests/conftest.py` with small seeded fixture tables built in `tmp_path`: normal, skewed, 2-group, k-group, contingency tables, a time series, transactions, and a causal setup.
  - Routing tests: one per row of `docs/association-tests.md` (Pearson, Spearman, Welch, ANOVA, Mann-Whitney, Kruskal-Wallis, chi-square, Fisher, degenerate). Each asserts `method` and `metadata.assumption_checks`.
  - Decline tests for every documented decline:
    - association with fewer than 10 rows
    - training with fewer than 30 rows, a single class, or a singleton class
    - forecast with fewer than 12 points
    - decompose with under two cycles
    - causal with fewer than 50 rows, a constant treatment, or a failed refutation
    - RFM with fewer than 5 customers
    - market basket with no itemsets
  - Determinism test: run each analytic twice on the same data. `values` and `metadata` must be identical.
  - Persistence round-trip: train, reopen the session, then evaluate.
  - Remove the `sys.path.insert` hack in `tests/test_containment.py:8`. Rely on the installed package instead.
  - Target: at least 80% line coverage of `service/algorithms/`, measured with `pytest-cov` in CI.
- **Pointers:** `src/tabint/analysis/service/algorithms/*.py`, `src/tabint/analysis/service/validation/assumptions.py`, `docs/honesty-model.md` (decline list), `docs/association-tests.md`. Libraries: pytest, pytest-cov (MIT).

## B-006 `explain_metric` reports an in-sample score, and its caveat is wrong
- **Status:** pending
- **Type:** bug
- **Priority:** P1
- **Effort:** S
- **Why:** The `explained` R² or accuracy comes from `tree.score(Xe, y)` on the same rows the tree was fit on. That is exactly the "graded on the rows it learned from" failure the README promises to prevent. Trust also ignores how much of the metric is actually explained. Separately, the caveat says "arithmetic attribution (which components moved the number)", which describes a different method.
- **Scope / acceptance criteria:**
  - Score the tree with a fixed-seed held-out split, or with k-fold CV (`cross_val_score`, `KFold(shuffle=True, random_state=0)`).
  - Report both `explained_train` and `explained_holdout`, and record the evaluation scheme in `metadata`.
  - Cap trust at `low` when held-out R² is below 0.1 (or accuracy is no better than majority class), with the caveat "segments explain little of the metric".
  - Change the caveat to "Drivers show what co-varies with the metric in this data, not what causes it."
  - Test: on pure-noise features with a random target, `explained_holdout` is about 0 and trust is `low`.
  - Verify through MCP: on a noise table, `explain_metric` no longer reports a high "explained" value.
- **Pointers:** `src/tabint/analysis/service/algorithms/insights.py:98`, `:108-113`; `docs/tools/drivers-and-causal.md`.

## B-007 Training and prediction responses skip the honesty envelope
- **Status:** pending
- **Type:** bug
- **Priority:** P1
- **Effort:** S
- **Why:** When training succeeds, `train_classifier` and `train_regressor` return a plain dict with no `trust` and no `declined` field. `add_predictions` returns `unassessed` and silently writes in-sample predictions for training rows as if they were genuine forecasts. Both break the invariant that every result carries an honest `trust` block.
- **Scope / acceptance criteria:**
  - Make `_train` in `tools.py` return a `result_dict`-shaped object:
    - `method` is `hist_gradient_boosting_classifier` or the equivalent
    - `values` holds n_train, n_test, features and classes
    - `trust` is `model._trust`
  - Keep `model_name` at the top level for compatibility.
  - In `add_predictions`:
    - also write `<model>_in_training_set` (boolean, derived)
    - add a caveat that predictions on training rows are optimistic
    - set trust from `model._trust`
    - for classifiers, optionally write `<model>_proba`
  - Store the train/test row ids (`_ti_row`) on `TrainedModel` so the flag is exact.
  - Verify: through MCP, `train_classifier` returns `trust.level`, and `add_predictions` creates the flag column, which `run_sql` can see.
- **Pointers:** `src/tabint/analysis/tools.py:523-537`; `src/tabint/analysis/service/workspace.py:352-363`; `src/tabint/analysis/service/algorithms/supervised.py:216-298` (`train_test_split` drops `_ti_row`, so pass the index through).

## B-008 `evaluate` should compare against a naive baseline and report class imbalance
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** S
- **Why:** On a 95/5 churn table, 95% accuracy is worthless. Today `evaluate` reports weighted metrics with no baseline, so a useless model reads as good.
- **Scope / acceptance criteria:**
  - For classifiers, compute these on the same test split:
    - a `DummyClassifier(strategy="most_frequent")` baseline
    - balanced accuracy, per-class precision and recall, PR-AUC (binary), and Brier score / log loss
  - For regressors, compute a `DummyRegressor(mean)` baseline, plus MAPE when y > 0.
  - Add `values.lift_over_baseline`.
  - Rules:
    - If the model does not beat the baseline by a set margin (e.g. balanced-accuracy gain < 0.02 or R² < 0.05), set trust to `low` with the caveat "no better than always guessing X".
    - If the minority class is under 10%, add an imbalance caveat.
  - Record the baseline strategy in `metadata`.
  - Tests: a random-label dataset gives `low`; a clean separable dataset keeps `high`.
- **Pointers:** `src/tabint/analysis/service/algorithms/supervised.py:301-371`; sklearn `DummyClassifier`, `DummyRegressor`, `balanced_accuracy_score`, `average_precision_score` (BSD-3).

## B-009 Detect target leakage before training and driver analysis
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** S
- **Why:** `compute_feature` columns are marked as features, so a column derived from the target (`revenue*1.0`, `churned_flag`) passes straight into `train_*` and `explain_metric` and produces near-perfect "honest" scores. Leakage is the most common reason a model looks better than it is.
- **Scope / acceptance criteria:**
  - Before fitting, score each feature on its own against the target: |Spearman| for a numeric target, single-feature AUC or Cramér's V for a categorical one.
  - Thresholds:
    - any feature with |r| ≥ 0.98 or AUC ≥ 0.99 → caveat naming it, and trust capped at `low`
    - an exact deterministic mapping (target = f(feature)) → decline, unless the caller passes `allow_leaky=[...]`
  - Also flag features whose SQL expression in `compute_feature` references the target column. Store expressions in a registry for this.
  - Record `metadata.leakage_checks`.
  - Tests: a table with `y` and `y_copy = y*2` makes `train_regressor` decline or downgrade, naming `y_copy`.
- **Pointers:** `src/tabint/analysis/service/algorithms/supervised.py:_train`, `insights.py:explain_metric`, `feature_computation.py:407` (compute_feature), `db/ducktable.py` (derived-column registry pattern to copy for an expression registry).

## B-010 Harden `join`: validate `how`, quote identifiers, detect fan-out
- **Status:** pending
- **Type:** bug
- **Priority:** P1
- **Effort:** S
- **Why:** `how.upper()` is spliced into SQL without validation. Join and projection identifiers use raw `"…"` instead of `quote_ident`, so column names containing `"` break. And a one-to-many join silently inflates n, which the session model says the explicit join exists to prevent, yet `join` reports nothing about it.
- **Scope / acceptance criteria:**
  - Allow-list `how` to {left, inner}, plus right and full if wanted.
  - Use `quote_ident` everywhere in `_build_join_sql` and `_build_projection`.
  - After the join, compare the joined row count with the first table's row count. Return:
    - `n_rows_before`, `n_rows_after`, `fanout_ratio`
    - the edges used, with coverage
    - a `trust` and `caveats` block, e.g. "rows multiplied 3.2× — the unit of analysis is now order-line, not customer"
  - Add a `join` parameter `on=[{"left": "t.col", "right": "u.col"}]` for explicit keys when detection is wrong.
  - Tests:
    - `how="CROSS"` raises
    - a column named `a"b` joins correctly
    - customers←orders reports fanout > 1
- **Pointers:** `src/tabint/analysis/service/workspace.py:615-687`, `src/tabint/analysis/service/relationships.py:127-141`, `src/tabint/analysis/tools.py:146-151`.

## B-011 End-to-end MCP test harness over stdio
- **Status:** pending
- **Type:** engineering
- **Priority:** P1
- **Effort:** M
- **Why:** Nothing tests the real wire path: argument validation, JSON serialization of NaN, Decimal and dates, error surfacing, or reopening a session after a restart. Agents only ever see that path.
- **Scope / acceptance criteria:**
  - Add `tests/e2e/` using `mcp.client.stdio.stdio_client` and `ClientSession`. It spawns `python -m tabint.app.mcp_server --stdio` with `TABULAR_BASE=tmp`.
  - Scripted scenarios:
    - `create_session` → `profile` → `classify_as_nominal` → `analyze_association` → `train_classifier` → `evaluate` → `feature_importance` → `explain_prediction`
    - `join` across two CSVs
    - `forecast` on a monthly series
    - a decline scenario
    - a kill-and-restart of the server, then reopening by `session_key`
  - Assert every analytic response parses as strict JSON (`json.loads(..., parse_constant=raise)`) and has `trust.level` and `declined`.
  - Golden snapshots of `method` and `metadata` keys, via syrupy (MIT), to catch routing drift.
  - `list_tools` returns 45 tools, each with a non-empty description.
  - Runs in CI (B-003) in under about 3 minutes.
- **Pointers:** `src/tabint/app/mcp_server.py`, `src/tabint/shared/serialize.py`. The mcp 1.x client API is in the `mcp` package already installed.

## B-012 Read Excel, Parquet and JSON files (roadmap)
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** M
- **Why:** The README currently tells users to "Save as CSV first". Most real business data arrives as .xlsx or Parquet.
- **Scope / acceptance criteria:**
  - In `ducktable.load_csv`, which should become `load_file`, dispatch on extension:
    - `.csv`, `.tsv`, `.txt` → `read_csv_auto`
    - `.parquet` → `read_parquet`
    - `.json` / `.jsonl` / `.ndjson` → `read_json_auto`
    - `.xlsx` → `read_xlsx` from the DuckDB `excel` extension
  - Add an optional `sheet` argument to `create_session` and `add_table`. Default: every sheet becomes a table, named `<file>_<sheet>`.
  - Load the excel extension before `confine()`, because external access is disabled afterwards. Bundle or pre-install it at warm-up so first use needs no network.
  - For `.xls` (legacy), either decline with a clear message or fall back to pandas `read_excel` with the optional `xlrd`.
  - Update the staging logic in `tools._ingest_paths` (the `file_not_found` message currently says "Pass the path to a CSV file").
  - Tests: one fixture per format loads, with the right row count and dtypes.
  - Verify: through MCP, `create_session(["~/x.xlsx"])` lists one table per sheet.
  - Remove the "Save as CSV" note from the README and docs.
- **Pointers:** `src/tabint/analysis/db/ducktable.py:38-41`, `src/tabint/analysis/service/workspace.py:535-546`, `src/tabint/analysis/tools.py:32-63`. DuckDB `excel` extension (MIT), `read_xlsx`: https://duckdb.org/docs/guides/file_formats/excel_import

## B-013 Rework time-series preparation and forecasting
- **Status:** pending
- **Type:** bug
- **Priority:** P1
- **Effort:** M
- **Why:** There are three problems:
  - `_ordered_series` treats every row as one time step, so order-level data with many rows per day becomes a nonsense series.
  - `_infer_period` picks 12 for any series with at least 24 points, whatever the frequency (daily data gets period 12).
  - `forecast` always fits ARIMA(1,1,1) with no seasonality, no accuracy check, and no future timestamps.
- **Scope / acceptance criteria:**
  - Add a shared `_regularize(store, time_col, value_col, freq=None, agg="sum")`. It infers frequency with `pd.infer_freq`, or from the median gap after aggregating duplicate timestamps. It resamples to that frequency and fills gaps explicitly. Record `freq`, `agg`, `n_gaps_filled` and `duplicates_aggregated` in `metadata`.
  - Set the seasonal period from frequency (D→7, W→52, M→12, Q→4, H→24). If there is under two cycles, use no seasonality, with a caveat.
  - Choose the forecast model deterministically: AutoETS / AutoARIMA (statsforecast, Apache-2.0, or statsmodels ETS) against a seasonal-naive baseline, picked by a rolling-origin backtest (fixed folds). Report backtest MASE and sMAPE, and set trust to `low` if the model doesn't beat seasonal-naive.
  - Return `dates` for each forecast step.
  - `decompose` returns arrays with dates. Add a `max_points` cap with the full arrays written back as columns instead of returned in the payload.
  - Tests: a daily series with weekly seasonality gets period 7; transaction-level rows are aggregated; forecast `dates` are monotonic.
- **Pointers:** `src/tabint/analysis/service/algorithms/timeseries.py:119-189` and `:280-296`; `compare.py` and `detect_changepoints` should reuse `_regularize`. statsforecast (Apache-2.0), statsmodels ETS (BSD-3).

## B-014 Multiple-testing correction and ordinal-aware tests in association
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** S
- **Why:** `association_matrix` runs O(p²) tests and returns only effect sizes, with no p-values and no correction, so an agent "scanning for strong cells" will find false positives. `categorical_ordinal` columns are routed exactly like nominal ones, which wastes the order the user declared.
- **Scope / acceptance criteria:**
  - `association_matrix` returns `p_values` and `q_values`, using Benjamini–Hochberg via `statsmodels.stats.multitest.multipletests`. Add `significant_after_fdr`, and a caveat with the number of tests run.
  - New routes in `analyze_association`, recorded in `metadata` and added to `docs/association-tests.md`:
    - ordinal × continuous → Spearman or Kendall τ-b, plus Jonckheere–Terpstra trend when there are at least 3 levels
    - ordinal × ordinal → Kendall τ-b
  - Use the ordinal level order the user supplies, by adding an optional `order` field to `set_column_type`. Without it, sort the values.
  - Add bootstrap confidence intervals for the effect size: a fixed seed, 1000 resamples, capped at a row sample.
  - Tests: on 20 noise columns, uncorrected p<0.05 cells exist but `significant_after_fdr` is almost always empty. The ordinal route is chosen when expected.
- **Pointers:** `src/tabint/analysis/service/algorithms/association.py:53-104`, `descriptive.py:178-237`, `validation/dtypes.py`, `tools.py:269-289`. statsmodels (BSD-3), `scipy.stats.kendalltau` (BSD-3).

## B-015 Stop pulling whole tables into Python
- **Status:** pending
- **Type:** engineering
- **Priority:** P1
- **Effort:** M
- **Why:** The code is "in-database by default" in its docs but not in practice:
  - `run_sql` materializes the full result and then truncates it.
  - `join`, `create_table` and `insert_into` call `get_frame()` just to count rows.
  - `association_matrix` re-materializes the whole table for every column pair.
  - `silhouette_score` is O(n²) memory.
  - `apriori` builds a dense one-hot matrix.

  Medium-sized files (1M+ rows) stall or run out of memory.
- **Scope / acceptance criteria:**
  - `run_sql`: wrap as `SELECT * FROM (<q>) LIMIT limit+1` and get the total with a separate `COUNT(*)` only when needed.
  - Use `Table.count_rows()` in `tools.py:151,192,208`.
  - `association_matrix`: materialize one frame, or only the needed columns, and pass it to the pair functions.
  - `profile`: compute stats in DuckDB (`SUMMARIZE`, or per-column aggregates) instead of pandas.
  - `list_categorical_columns`: use a SQL `DISTINCT … LIMIT 10`.
  - `silhouette_score(..., sample_size=min(n, 10_000), random_state=0)`.
  - `market_basket`: use `fpgrowth` with a sparse `TransactionEncoder` output.
  - Add a benchmark script `scripts/bench.py` on a synthetic 1M-row table that records per-tool time and peak RSS. Document the targets, e.g. profile under 5 s and association_matrix (10 columns) under 30 s.
- **Pointers:** `src/tabint/analysis/tools.py:155-161`, `:150-151`, `:191`, `:208`, `:255-266`; `algorithms/descriptive.py:33`, `:210`, `:220`; `association.py:94`; `clustering.py:160-165`; `basket.py:60-72`.

## B-016 Keep long-running tools from blocking the server, and serialize access per session
- **Status:** pending
- **Type:** engineering
- **Priority:** P1
- **Effort:** M
- **Why:** In mcp 1.29, FastMCP calls synchronous tool functions directly on the event loop. Training, SHAP, t-SNE or a DoWhy run freezes the whole HTTP server, including pings, and clients time out. Two agents running at once (Claude Desktop and Claude Code each spawn a stdio server) hit DuckDB file-lock errors on the same session. Separately, there is unused slow-lane job scaffolding (`service/jobs/`).
- **Scope / acceptance criteria:**
  - Make tools `async` and run the work in `anyio.to_thread.run_sync`.
  - Add a per-`session_key` `threading.Lock`, so concurrent calls on one session are serialized and calls on different sessions run in parallel.
  - Use `ctx.report_progress` for the k-sweep in clustering, permutation repeats and backtest folds.
  - Detect the DuckDB lock conflict ("Could not set lock on file") and return a clear `session_locked` error naming the other process. Optionally open read-only for read tools.
  - Either wire up `jobs/` as `start_job`, `job_status` and `job_result` tools for operations whose estimated cost is above a threshold, or delete it.
  - Test: two concurrent `train_classifier` calls on different sessions overlap in time, and a `ping` during training answers in under 1 s.
- **Pointers:** `src/tabint/analysis/tools.py` (all tools), `src/tabint/shared/server.py:44-55`, `src/tabint/analysis/service/jobs/{registry,runner}.py`, `.venv/.../mcp/server/fastmcp/utilities/func_metadata.py:92-95`. The MCP Tasks extension (2026-07-28 spec) is the long-term target (see B-032).

## B-017 Tool annotations, output schemas and one error format
- **Status:** pending
- **Type:** engineering
- **Priority:** P1
- **Effort:** S
- **Why:** Clients can't tell read-only tools from tools that change tables, results have no machine-readable schema, and errors come in three shapes: a raised `ValueError`/`KeyError` text, `{"ok": false, "error": …}` dicts, and `declined` results. Agents handle each one differently.
- **Scope / acceptance criteria:**
  - Add `ToolAnnotations`:
    - `readOnlyHint=True` on profile, association, count, relationships, run_sql, evaluate and the explain tools
    - `destructiveHint=False, idempotentHint=True` on the write-back feature tools
    - `openWorldHint=False` everywhere
  - Return a Pydantic `AnalysisResult` model (method, summary, trust, declined, values, metadata), so FastMCP emits an `outputSchema` and `structuredContent`.
  - Wrap every tool in a decorator that turns known exceptions into `{"ok": false, "error": "<code>", "message": …, "hint": …}`. Codes: unknown_session, unknown_table, unknown_column, unclassified_columns, invalid_argument, missing_dependency.
  - Keep `declined` as a successful result, not an error.
  - Tests in the e2e harness (B-011): `list_tools` shows annotations, and a bad column name returns `error == "unknown_column"`.
- **Pointers:** `src/tabint/analysis/tools.py`, `src/tabint/shared/serialize.py`, `src/tabint/shared/results.py`. MCP Python SDK 1.x structured output: https://py.sdk.modelcontextprotocol.io/servers/structured-output/

## B-018 A/B test (experiment) analysis tool
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** M
- **Why:** "Did variant B win?" is one of the most common business questions. It needs a sample-ratio-mismatch (SRM) check, the right test for proportions or means, variance reduction, power, and multiple-metric correction. Agents usually get these wrong when they improvise.
- **Scope / acceptance criteria:**
  - New tool `analyze_experiment(session_key, table, variant_column, metrics: list[str], control=None, covariate=None, alpha=0.05)`.
  - Route deterministically by metric type:
    - binary → two-proportion z-test, or Fisher for small counts
    - continuous → Welch t-test, or Mann-Whitney / bootstrap when heavily skewed
    - ratio metrics → delta method
  - Use CUPED when a pre-period `covariate` is given.
  - Always run an SRM chi-square. If SRM fails (p<0.001), decline.
  - Report per metric: absolute and relative lift with CI, p-value, Holm or BH correction across metrics, and achieved power plus minimum detectable effect.
  - Trust: `low` if underpowered (<0.5) or there are fewer than 100 units per arm.
  - Add a companion `power_analysis` tool for sample-size planning.
  - Document the routing in `docs/tools/` and the association-tests style doc.
  - Tests: seeded simulated experiments with known lift; an SRM fixture declines.
- **Pointers:** new `src/tabint/analysis/service/algorithms/experiment.py`, wired through `workspace.py` → `session.py` → `tools.py`. tea-tasting (MIT, works natively on ibis and DuckDB): https://github.com/e10v/tea-tasting. Or statsmodels `proportions_ztest` and `stats.power` (BSD-3) with no new dependency.

## B-019 Multivariate anomaly detection
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** M
- **Why:** `detect_outliers` only checks one column at a time with IQR and z-score. "Which orders or customers look unusual?" needs row-level scoring across many columns (fraud, data-entry errors).
- **Scope / acceptance criteria:**
  - New tool `detect_anomalies(session_key, table, columns=None, contamination="auto")`. Use the shared `_prep.numeric_matrix`.
  - Pick the method deterministically:
    - IsolationForest(random_state=0) by default
    - LocalOutlierFactor when there are fewer than 1000 rows and fewer than 20 features
  - Write back `anomaly_score` and `is_anomaly`, both derived (`feature=False`).
  - Return the top-k anomalous rows with the features that pushed each score up (SHAP on IsolationForest, or per-feature z-contributions).
  - Trust:
    - from sample size
    - a caveat that anomalies are statistical, not confirmed errors
    - `low` if the score distribution is unimodal with no clear separation (e.g. a gap statistic between flagged and unflagged scores)
  - Tests: planted anomalies in a seeded dataset are ranked in the top 5%.
- **Pointers:** new function in `algorithms/descriptive.py` or a new `anomaly.py`; `_prep.py:154-186`. scikit-learn IsolationForest and LOF (BSD-3); PyOD (BSD-2) as an optional extra for ECOD/COPOD.

## B-020 Prediction intervals and calibrated probabilities (conformal)
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** M
- **Why:** A point prediction with no uncertainty is a "confident-looking number by omission". Conformal intervals give distribution-free coverage guarantees, which suits the honesty model well.
- **Scope / acceptance criteria:**
  - When training, keep a calibration split carved deterministically from the training portion.
  - Regressors: `add_predictions` also writes `<col>_lower` and `<col>_upper` at 90% coverage (configurable), using split-conformal or CV+ residual quantiles.
  - Classifiers: write calibrated `<col>_proba`, via `CalibratedClassifierCV` (isotonic when there are 1000+ calibration rows, sigmoid otherwise), plus an optional conformal prediction set.
  - `evaluate` reports empirical coverage on the test split, and a calibration error (ECE / Brier).
  - Trust is downgraded when empirical coverage falls more than 5 points below nominal.
  - Record the method, alpha and calibration size in `metadata`.
  - Tests: on seeded data, coverage on the test split is within ±5 points of nominal.
- **Pointers:** `src/tabint/analysis/service/algorithms/supervised.py` (`TrainedModel`, `_train`, `evaluate`), `workspace.py:352-363`. MAPIE (BSD-3) or crepes (BSD-3), or about 40 lines of hand-written split-conformal with no dependency; sklearn `CalibratedClassifierCV` (BSD-3).

## B-021 Survival / time-to-event analysis
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** M
- **Why:** "How long until customers churn?" and "does the premium tier keep customers longer?" are core retention questions. Treating censored customers as churned or as retained gives the wrong answer either way, and `retention_cohorts` is descriptive only.
- **Scope / acceptance criteria:**
  - New tool `survival(session_key, table, duration_column | (start_column, end_column), event_column, group_column=None, covariates=None)`.
  - Return:
    - a Kaplan–Meier curve (downsampled points), median survival with CI, and survival at standard horizons
    - with `group_column`: log-rank test (2 groups) or multivariate log-rank (k groups)
    - with `covariates`: a Cox PH model with hazard ratios, CIs and concordance, plus a Schoenfeld-residual proportional-hazards check. If PH is violated, add a caveat and set trust to `low`.
  - Declines:
    - fewer than 10 events
    - censoring above 95%
    - an event column that isn't binary
  - Record the censoring rate in `metadata.basis`.
  - Tests against lifelines' bundled `load_waltons` / `load_rossi` reference values.
- **Pointers:** new `algorithms/survival.py`. lifelines (MIT); add it to the `insights` extra. Avoid scikit-survival, which is GPL-3.0.

## B-022 Handle large files with deterministic sampling (roadmap)
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** L
- **Why:** The roadmap promises that "large data degrades gracefully rather than being refused". Today every model-based tool loads the full table into pandas.
- **Scope / acceptance criteria:**
  - Add a per-tool row budget, configurable through `MEELU_MAX_MODEL_ROWS` (default e.g. 200k).
  - Above the budget, use `USING SAMPLE reservoir(N ROWS) REPEATABLE(<seed>)` in DuckDB. For classification, stratify on the target.
  - Record `metadata.sampling = {"method", "n_sampled", "n_total", "seed"}` and a trust caveat. Never take trust above `moderate` when sampled, unless a stability check (re-run on a second seed agrees within a tolerance) passes.
  - Keep descriptive tools fully in-database on the full data (depends on B-015).
  - Use streaming CSV ingest. DuckDB already streams, so mainly verify memory at 5 GB.
  - Add a documented ceiling with a clear decline above it.
  - Write-back on a sample: either predict across the full table in batches (for models), or label only the sampled rows with an explicit NULL meaning "not scored" (clustering), and document which.
  - Benchmark: a 20M-row synthetic table profiles in under 60 s, and `train_classifier` finishes within the memory budget.
- **Pointers:** `src/tabint/analysis/service/_prep.py:36-38` (`get_frame`, the central place to add a sample), `db/ducktable.py:293-297`, every algorithm that calls `store.get_frame()`.

## B-023 Bundle of small correctness fixes
- **Status:** pending
- **Type:** bug
- **Priority:** P1
- **Effort:** S
- **Why:** Each of these is a small fix, but each one can give an agent a wrong or confusing answer today.
- **Scope / acceptance criteria:**
  - **Multiclass SHAP explains the wrong class.** It picks the class with the largest absolute contribution instead of the predicted class. Explain the predicted class, and return `explained_class`. `interpretation.py:119-123`.
  - **`explain_prediction` has no bounds check.** Negative `row_index` silently wraps around. Validate `0 <= row_index < n`, and return `in_training_set` (after B-007). `tools.py:559-563`.
  - **`cluster` overwrites a user's own column** named `cluster`. Use `cluster_<k>`, or refuse when a non-derived column has that name. `clustering.py:22`, `:70`.
  - **`cluster` uses PCA, t-SNE and UMAP components as features** alongside the raw columns (they are written with `feature=True`), so it double-counts. Exclude them by default. `clustering.py:44`, `dimreduction.py:72`.
  - **`reduce_dimensions` excludes any user column whose name starts with `pca_`, `tsne_` or `umap_`.** Use the derived registry instead of name prefixes. `dimreduction.py:42-45`.
  - **`rfm` doesn't save segments back**, even though the README says "Segments… get saved back". Write a per-customer RFM table, e.g. `<table>_rfm`, through `create_table`. `cohort.py:84-147`.
  - **Thin-data cases raise exceptions instead of declining.** Fix in `compare_periods` (each side fewer than 2 points, `compare.py:52-55`), `decompose` and `detect_changepoints` (fewer than 4 points, `timeseries.py:34-37`, `:226-229`), and in `_categorical_continuous` (k<2, `association.py:186-189`).
  - Each fix gets a regression test.
- **Pointers:** the files and lines listed above.

## B-024 Make causal_effect more rigorous
- **Status:** pending
- **Type:** feature
- **Priority:** P1
- **Effort:** M
- **Why:** By default `causal_effect` adjusts for every other column, including post-treatment variables and mediators, which biases the estimate. The only refutation is `random_common_cause` with a loose 50% tolerance, though the docs call it a "placebo refutation". There is no overlap (positivity) check, and categorical confounders go into a linear regression unencoded.
- **Scope / acceptance criteria:**
  - Require explicit `confounders`. Alternatively keep the default but add a caveat and cap trust at `low` when they are defaulted. Also add an optional `exclude_post_treatment=[...]`.
  - Run three refuters and require all to pass: `placebo_treatment_refuter` (permute), `random_common_cause` and `data_subset_refuter`. Record each one's result.
  - For a binary treatment: fit a propensity model, report overlap (share of units with propensity outside [0.05, 0.95]), and decline when overlap is poor.
  - One-hot encode nominal confounders before passing them to DoWhy.
  - Optional nonlinear estimator, chosen deterministically: DoubleML PLR with gradient-boosted nuisances when n ≥ 1000. Otherwise linear. Record the choice.
  - Fix `docs/honesty-model.md` and `docs/tools/drivers-and-causal.md` to name the refuters accurately.
  - Tests: on a simulated DGP with a known effect and a mediator, defaulting confounders is flagged; the explicit set recovers the effect within its CI; a no-overlap fixture declines.
- **Pointers:** `src/tabint/analysis/service/algorithms/causal.py:69-131`. DoWhy (MIT), DoubleML (BSD-3), EconML (MIT).

## B-025 Audit trust coverage and enforce it with a test
- **Status:** pending
- **Type:** engineering
- **Priority:** P1
- **Effort:** S
- **Why:** The roadmap item "Better trust ratings — real confidence assessments everywhere they're still missing" has no inventory and nothing that enforces it. Some results are `unassessed` (`add_predictions`), and `profile` is always `high`, even for a 3-row table.
- **Scope / acceptance criteria:**
  - Write a test that calls every analytic tool on a fixture table and asserts `trust.level != "unassessed"`, except for an explicit allow-list with a reason for each entry.
  - Give `profile` a sample-size floor and caveats for type ambiguity, e.g. numeric-looking strings or an ID column classified as continuous.
  - Include `basis` on every result.
  - Add a "Trust by tool" table to `docs/honesty-model.md` saying what drives each tool's level.
  - Make the server instructions mention that structural tools (`run_sql`, `join`) carry no trust block, or give them one.
- **Pointers:** `src/tabint/analysis/service/algorithms/descriptive.py:79-91`, `src/tabint/analysis/service/workspace.py:358-363`, `src/tabint/shared/honesty.py`, `docs/honesty-model.md`.

## B-026 Windows support (roadmap)
- **Status:** pending
- **Type:** engineering
- **Priority:** P2
- **Effort:** M
- **Why:** The installer is POSIX `sh` only. `install.py` already has `IS_WIN` and `APPDATA` paths, but they are untested. Windows is a large share of spreadsheet users.
- **Scope / acceptance criteria:**
  - Add `install.ps1`: install uv with `irm https://astral.sh/uv/install.ps1 | iex`, then run `scripts/install.py`.
  - Check every agent's config path on Windows: Claude Desktop, Cursor, VS Code, Codex, Zed.
  - Handle the `.exe` binary name and backslash paths in DuckDB `allowed_directories`, escaping them in `confine()`.
  - Make staging cleanup tolerate Windows file locks.
  - Add windows-latest to the CI matrix (B-003) and pass both the unit and e2e suites.
  - Update the README "Getting set up" with a PowerShell one-liner.
- **Pointers:** `install.sh`, `scripts/install.py:35-37`, `:270-285`; `src/tabint/analysis/service/workspace.py:440-468`; `src/tabint/analysis/tools.py:66-86`.

## B-027 Analyse related tables without a manual join (roadmap)
- **Status:** pending
- **Type:** feature
- **Priority:** P2
- **Effort:** M
- **Why:** The roadmap lists "Fewer manual steps". Agents often forget to join, or join at the wrong grain. The one-table rule should remain, but the engine can build the table itself.
- **Scope / acceptance criteria:**
  - New tool `prepare_analysis_table(session_key, target_table, columns: list["table.column"], grain="target_table")`.
    - Walk the FK graph from `target_table`.
    - Left-join parent tables (many-to-one, which is safe).
    - For child tables (one-to-many), aggregate to the target grain with deterministic default aggregates (count, sum, mean, last) instead of fanning out.
    - Record the plan in `metadata.join_plan` and report the fan-out check from B-010.
  - Analytic tools keep requiring one table. Error messages for a column that lives in another table should point to this tool.
  - Tests: orders + customers + order_items produce one row per customer, with item aggregates.
- **Pointers:** `src/tabint/analysis/service/workspace.py:615-687`, `src/tabint/analysis/service/relationships.py`, `docs/session-model.md` (one-table rule).

## B-028 Data-quality check and table drift comparison
- **Status:** pending
- **Type:** feature
- **Priority:** P2
- **Effort:** S
- **Why:** Before trusting any analysis, users need to know about duplicates, mixed types, impossible values and stale data. "Has this month's extract shifted from last month's?" is a common recurring question.
- **Scope / acceptance criteria:**
  - New tool `check_quality(table)`, computed in DuckDB, reporting:
    - exact-duplicate rows and duplicate keys on identifier columns
    - constant and near-constant columns, and numeric-as-string columns
    - negative values in columns named like amount, qty or price
    - future dates and date-range gaps
    - high-cardinality categoricals and columns that are mostly null
    - a severity per finding
  - New tool `compare_tables(table_a, table_b)`: per-column drift, using PSI or KS for numeric columns and chi-square or Jensen-Shannon for categorical ones, with BH correction (B-014) and a flag for schema differences.
  - Both get trust blocks. Thresholds are recorded in `metadata`.
- **Pointers:** new `algorithms/quality.py`. scipy/statsmodels only. Evidently (Apache-2.0) is optional but heavy, so not recommended.

## B-029 Basic support for free-text columns
- **Status:** pending
- **Type:** feature
- **Priority:** P2
- **Effort:** M
- **Why:** Free-text columns (reviews, ticket descriptions, notes) are currently classified as `identifier` because their distinct ratio is above 0.9, and are silently dropped from every analysis.
- **Scope / acceptance criteria:**
  - Add a `text` type to `COLUMN_TYPES`, assigned deterministically: a string column with mean token count ≥ 5 and distinct ratio > 0.5. Show it in `profile`.
  - New tool `analyze_text(table, column, n_topics=None)`:
    - top terms
    - TF-IDF + NMF topics (k by a fixed rule or coherence, seeded)
    - per-row topic written back
    - optionally, keyword association with a target column via chi-square with FDR
  - Optional `text_features` tool: length, word count and topic weights as model-eligible columns.
  - Optional embeddings behind an extra (model2vec, MIT, CPU-only, small). They must be pinned to a fixed model version for determinism.
  - Tests: a seeded review corpus recovers the planted topics.
- **Pointers:** `src/tabint/analysis/service/validation/dtypes.py:125-131`, `_prep.py`. scikit-learn TfidfVectorizer and NMF (BSD-3), model2vec (MIT).

## B-030 Cross-validated metrics and optional extra model backends
- **Status:** pending
- **Type:** feature
- **Priority:** P2
- **Effort:** M
- **Why:** A single 75/25 split gives noisy metrics on small tables, and there is no time-aware split for data with dates. LightGBM, XGBoost and CatBoost bring native categoricals and monotonic constraints, but HistGradientBoosting is already strong, so they are lower value.
- **Scope / acceptance criteria:**
  - `evaluate` adds stratified 5-fold CV mean ± std (fixed seed) when n < 5000.
  - If the table has a datetime column and the caller passes `time_column`, use a forward-chaining split (TimeSeriesSplit) and record it.
  - Optional backends `lightgbm` / `catboost` behind an extra, with fixed seeds and `deterministic=True` (LightGBM).
  - Optional `backend="auto"`: FLAML (MIT) with `max_iter` (not a time budget) and a seed, for reproducibility. It records the chosen learner and hyperparameters in `metadata`. Leave the TabPFN/TabICL lane untouched; it is being handled separately.
  - Tests: CV std is reported; the same seed gives the same metrics across runs.
- **Pointers:** `src/tabint/analysis/service/algorithms/supervised.py:62-90`, `:275-279`. LightGBM (MIT), CatBoost (Apache-2.0), XGBoost (Apache-2.0), FLAML (MIT).

## B-031 Session lifecycle tools and safer model storage
- **Status:** pending
- **Type:** engineering
- **Priority:** P2
- **Effort:** M
- **Why:** There is no way to delete sessions or tables, so `$TABULAR_BASE` grows forever. Models are stored as pickles that are skipped silently if they fail to load after a scikit-learn upgrade (`except Exception: continue`). Agents also lack a cheap "describe table" call.
- **Scope / acceptance criteria:**
  - New tools:
    - `delete_session(session_key)`, which also closes the connection and removes it from `_SESSIONS`
    - `drop_table(session_key, table)`
    - `describe_table`: schema, cached types, derived columns, row count, all without loading the table
  - `list_sessions` returns created and updated times, table names and size on disk, from `meta.json`.
  - Save `sklearn.__version__` and the package version with each model. On load, report models that failed to load in `session_info` (`stale_models`) instead of dropping them silently.
  - Note in the docs that pickles under `$TABULAR_BASE` are trusted code. Load only from the session's own directory.
  - On startup, sweep orphaned `.staging/*` directories older than 1 hour.
  - Tests for each tool, and a corrupt-pickle case.
- **Pointers:** `src/tabint/analysis/db/persistence.py:112-147`, `src/tabint/shared/server.py:41-55`, `src/tabint/analysis/tools.py:112-123`.

## B-032 Packaging and dependency hygiene, plus a plan for MCP SDK v2
- **Status:** pending
- **Type:** engineering
- **Priority:** P2
- **Effort:** S
- **Why:** Several small issues add up:
  - The server advertises itself as `"tabint"`.
  - The version number is defined in three places.
  - Dependencies have no upper bounds (the venv already resolved pandas 3.0 and scikit-learn 1.9), and `uv tool install` ignores `uv.lock`.
  - mcp is pinned `<2`, but SDK v2 targets the 2026-07-28 stateless spec.
- **Scope / acceptance criteria:**
  - `FastMCP("meelu-analytics", …)`.
  - Single-source the version through `importlib.metadata`. Have the publish workflow check that the tag, `pyproject` and `server.json` agree.
  - Add tested upper bounds (`pandas<4`, `scikit-learn<2`, `duckdb<2`, `ibis-framework<13`), with Dependabot for bumps.
  - Remove the stale "Apache-2.0" and "~30 tools" comments in the `pyproject.toml` header and `tools.py:1`.
  - Fix the description of the `server.json` `TABULAR_BASE` variable.
  - Write a short spike document on migrating to mcp v2: tasks extension for long jobs, the `input_required` flow for choosing a target column. List breaking changes, but don't migrate yet.
- **Pointers:** `src/tabint/shared/server.py:38`, `pyproject.toml:1-68`, `server.json`, `src/tabint/analysis/tools.py:1-2`, `.github/workflows/publish-mcp.yml:402-407`.

## B-033 Optional authentication on the HTTP transport
- **Status:** pending
- **Type:** engineering
- **Priority:** P2
- **Effort:** S
- **Why:** The README suggests exposing the HTTP server for Claude on the web and Cowork, but there is no authentication. Anyone who can reach the port can ingest any file the process can read (ingest ignores the sandbox by design) and then read it back with `run_sql`.
- **Scope / acceptance criteria:**
  - Add a `MEELU_ANALYTICS_TOKEN` env var. When it is set, require `Authorization: Bearer <token>` through Starlette middleware on the streamable-HTTP app.
  - Refuse to bind to a non-loopback host without a token, unless `MEELU_ALLOW_INSECURE=1` is set.
  - Add an option to restrict ingest to the data root (`MEELU_INGEST_ANYWHERE=0`) for exposed deployments.
  - Document all of this in `docs/configuration.md`.
  - Tests: a request without the token gets 401; non-loopback binding without a token exits non-zero.
- **Pointers:** `src/tabint/app/mcp_server.py:24-35`, `src/tabint/analysis/tools.py:32-63`, `docs/configuration.md:15-21`, `README.md:147-153`.

## B-034 Fix documentation drift and add contributor and testing docs
- **Status:** pending
- **Type:** docs
- **Priority:** P2
- **Effort:** S
- **Why:** The docs are the contract with agents and contributors, and several pages no longer match the code.
- **Scope / acceptance criteria:**
  - Correct these mismatches:
    - the causal refuter is described as a "placebo refutation" but is actually random-common-cause (until B-024 lands)
    - code comments say "~30 tools" and "all 44 tools"; there are 45
    - `TABULAR_BASE` is described as "the only directory the server may read from"
    - install hints name the nonexistent `tabint[...]` package
  - Add a `CONTRIBUTING.md` section on running the unit and e2e suites (B-005, B-011), adding a routing row to `docs/association-tests.md`, and the trust-coverage test (B-025).
  - Add a "Limits" page: max sensible rows per tool, what declines and when, determinism guarantees (seeds), and which tools write columns back.
  - Write the docs for the new tools at the same time as the features land.
- **Pointers:** `docs/honesty-model.md:72`, `docs/tools/drivers-and-causal.md:31`, `docs/configuration.md`, `src/tabint/shared/honesty.py:11`, `src/tabint/analysis/tools.py:1`, `README.md:270-281`.
