# Supervised machine learning

Train a model on a table, evaluate it honestly, and explain what it learned.

The whole family shares one design decision: **the held-out split is real and it
is used everywhere**. Training does a 75/25 split, stratified for classification.
Evaluation, permutation importance, and every reported metric come from the 25%
the model never saw. A metric computed on training data is not a metric, it is a
memory test.

## `train_classifier(session_key, table, target, name=None, backend="gbt")`
## `train_regressor(session_key, table, target, name=None, backend="gbt")`

Train a model to predict `target` from every other usable column, and persist it
under `name` (defaulting to the target's name). Models are saved to the session,
so they survive a server restart.

A scikit-learn `Pipeline` handles preprocessing — imputation, plus one-hot or
ordinal encoding of categoricals — so the fitted transform travels with the
model and is applied identically at prediction time. That is what makes
`add_predictions` and `explain_prediction` safe to call later.

**Backends:**

| Backend | What it is |
|---|---|
| `gbt` (default) | `HistGradientBoosting` — fast, strong on tabular data, no GPU |
| `tabicl` | TabICL v2, a tabular foundation model. No per-task training; strong on small and medium tables. Needs `uv add tabicl`, GPU recommended |

**Training refuses** under 30 usable rows, or when a class has fewer than 2
examples. A model whose metrics are meaningless is worse than a refusal — it
produces a number that looks like evidence. The refusal comes back as a declined
result and nothing is saved. See the [Honesty model](../honesty-model.md).

Returns the model name, target, task, backend, and the feature columns used.

## `evaluate(session_key, table, model_name)`

Metrics on the held-out split.

| Task | Metrics |
|---|---|
| Classification | accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| Regression | MAE, RMSE, R² |

Read the confusion matrix, not just the accuracy. On imbalanced data — 95% of
rows in one class — a model that predicts the majority every time scores 95% and
is worthless, and only the matrix shows it. ROC-AUC is the more honest headline
number there.

MAE and RMSE are both reported because they disagree usefully: RMSE punishes
large errors quadratically, so a gap between the two means your errors are
concentrated in a few bad predictions rather than spread evenly.

## `feature_importance(session_key, table, model_name)`

Permutation importance — shuffle one column, measure how much held-out
performance drops — with 10 repeats, aggregated back to the original columns.

Aggregation matters: a categorical column becomes many one-hot columns during
training, and importance reported per dummy is unreadable. These are summed back
so you get one number per column you actually have.

Permutation importance is model-agnostic and measured on held-out data, which
makes it far more trustworthy than a tree model's built-in impurity importance
(which is computed on training data and biased toward high-cardinality columns).
Its known weakness is correlated features: if two columns carry the same
information, shuffling either one alone barely hurts, and both look unimportant.

## `add_predictions(session_key, table, model_name, column_name=None)`

Write the model's predictions back as a column, for every row in the table.

Because the fitted preprocessing pipeline is stored with the model, the same
transforms are applied automatically. The new column is a normal table column —
queryable with `run_sql`, comparable against the actual values, usable as a
feature.

Note that predictions for rows in the training portion are optimistic. Use
`evaluate` for the honest performance number.

## `explain_prediction(session_key, table, model_name, row_index=0)`

A SHAP local explanation for one row (0-based index): which features pushed this
particular prediction up or down, and by how much. Uses the exact `TreeExplainer`
for the gradient-boosted default, and contributions are summed back to the
original columns like importances are.

This answers a different question from `feature_importance`. Importance is global
— what the model relies on across all rows. SHAP here is local — why *this* row
got *this* prediction. A feature that barely matters overall can dominate a single
case, which is usually the interesting one.

SHAP explains the model, not the world. A contribution is a statement about how
this fitted model responds to that feature, not evidence that the feature causes
the outcome. For that question, see [`causal_effect`](drivers-and-causal.md).
