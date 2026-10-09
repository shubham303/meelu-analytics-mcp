# Tabular foundation model

An optional pre-trained model that `train_classifier` and `train_regressor` use
automatically once it is installed, one tool to install it on demand, and one
tool to fine-tune it.

A *tabular foundation model* is a transformer pre-trained on millions of
synthetic tables. Given your training rows as context, it predicts new rows in a
single forward pass. There is no per-task training. On small and medium tables
it usually matches or beats tuned gradient boosting with no tuning at all. That
makes it the right default for questions like *"which of these customers will
churn?"* on a few thousand rows.

This server uses **TabICL v2** ([paper](https://arxiv.org/abs/2602.11139),
[code](https://github.com/soda-inria/tabicl)), installed only when it is asked
for.

## Installing it: `install_foundation_model(wait_seconds=0)`

PyTorch and TabICL are several hundred MB, so they are **never** part of the
server's install — that stays fast. Instead:

1. `train_classifier` / `train_regressor` with the default `backend="auto"` train
   gradient-boosted trees as usual. When the table is one the foundation model
   would have handled, the response carries a `hint`: the model isn't
   installed, call `install_foundation_model`, then retrain.
2. `install_foundation_model` starts a **background** install and returns
   immediately with `state: "installing"`. Calling it again reports progress
   (`step`), and finally `"ready"` — or `"failed"` with a `reason` (calling again
   retries). `wait_seconds` (max 120) blocks that long first, so an agent can
   poll without spamming.
3. Once `"ready"`, retrain: `auto` now picks TabICL.

What the install does:

| Step | Detail |
|---|---|
| Packages | `uv pip install --target <TABULAR_BASE>/.deps/foundation` of `tabicl>=2.2,<2.3`, `transformers` (fine-tuning only) and `torch` — `python -m pip` if `uv` isn't on PATH. CPU-only torch: the PyTorch CPU index on Linux (the default Linux wheel bundles GBs of CUDA), the default wheels on macOS. Versions are constrained to the running numpy / scipy / scikit-learn / pandas and resolved for the running Python. About 700 MB on disk. |
| Location | Outside the package's own environment, so it survives `uv tool install --force` upgrades and `uvx` cache wipes. Packages land in a temporary sibling folder that is renamed into place only on success, so a failed install never leaves a half-usable folder. |
| `sys.path` | The folder is **appended** lazily, only on the foundation code path (and before saved models are unpickled), so the core environment's own libraries always win. |
| Weights | Both pinned checkpoints (~110 MB each) download into the Hugging Face cache (`~/.cache/huggingface`). After that the model works offline. |

"Installed" means *packages importable and weights cached*, so a training call
never starts a multi-hundred-MB download itself. With
`MEELU_FOUNDATION_AUTO_INSTALL=1` the server starts the background install on its
own the first time `auto` would have used the model (that call still trains
trees). People who want it preinstalled can use the `foundation` extra
(`uv sync --extra foundation`); the weights are still fetched by
`install_foundation_model`.

Measured from an environment without torch (macOS, warm `uv` cache): the tool
returned in under a second; packages were in place within a minute; the weights
download took ~2.5 minutes on a slow connection; the first TabICL training after
that took ~29 s (torch import and model load), later ones ~9 s.

## How the model is chosen

The MCP tools `train_classifier` and `train_regressor` take `backend="auto"` by
default. (The Python API, `Session.table(...).train_classifier`, keeps `"gbt"` as
its default; pass `backend="auto"` there to opt in.) The rule depends only on
whether the model is installed and on the table's shape — both recorded — so the
same table on the same install gets the same choice:

| Check | Outcome when it fails |
|---|---|
| Foundation model installed (packages + weights) | gradient-boosted trees (`gbt`), plus a `hint` if the table is within the envelope |
| ≤ 10,000 usable rows (CPU budget) | `gbt` |
| ≤ 500 features after one-hot encoding | `gbt` |
| ≤ 10 classes (classification) | `gbt` |
| All pass | **TabICL v2** (`tabicl`) |

If TabICL is installed but then can't run (a corrupted cache, or its forward
pass runs out of memory, for example), `auto` falls back to `gbt`. The failure is
recorded in `backend_selection.foundation_error` and a caveat is added to the
trust block. The result is never a traceback.

Every training result records the decision in `metadata.backend_selection`:

```json
"backend_selection": {
  "requested": "auto",
  "chosen": "tabicl",
  "reason": "TabICL v2 installed and the table is within its envelope (≤10,000 rows, ≤500 features, ≤10 classes) — using TabICL v2.",
  "checks": {"foundation_installed": true, "n_rows": 600, "n_encoded_features": 6, "n_classes": 2, "...": "..."}
}
```

You can also choose the backend yourself:

- **`backend="gbt"`** always uses trees.
- **`backend="tabicl"`** forces the foundation model, up to 100,000 rows and
  2,000 features. If it isn't installed, it returns a **declined** result
  pointing at `install_foundation_model` rather than quietly substituting trees.

## Honest evaluation

The foundation model is scored exactly like any other model, on the same
stratified 75/25 split. The 25% held-out rows are never in its context.

The training response contains the held-out metrics and a `trust` block derived
from them:

| Trust input | Effect |
|---|---|
| Training rows | The usual sample-size floor. |
| Test rows | Same floor, applied to the held-out split. |
| No held-out skill | **Low** trust with a caveat. "No skill" means accuracy no better than always predicting the most common class, or R² ≤ 0. |
| Foundation model used | A caveat saying so. |

## Downstream tools

| Tool | With the foundation model |
|---|---|
| `evaluate` | Unchanged. Held-out accuracy / precision / recall / F1 / ROC-AUC, or MAE / RMSE / R². |
| `add_predictions` | One batched predict over the whole table. Rows the model trained on were in its context, so their predictions sit close to their known labels — a caveat says so; use `evaluate` for real accuracy. |
| `feature_importance` | Permutation importance, with **5** repeats instead of 10, on at most 1,000 held-out rows (a fixed-seed sample). Shuffled copies are scored in batched predict calls of up to 50,000 rows, because much of TabICL's cost is per call (it re-reads the training context). Recorded as `metadata.n_repeats` and `metadata.n_rows_scored`. |
| `explain_prediction` | **Approximate** permutation SHAP over a fixed 20-row background, with 2·features+1 evaluations. Returned as `method: "shap_permutation"`. For binary targets it explains the probability of the positive class (`metadata.explained_class`), like the trees backend. A caveat says the contribution sizes are estimates and their ranking is more reliable. |

On an 8 GB Apple-silicon laptop (CPU only, 4 torch threads), a 600-row churn table measured: train and score
~9 s, `evaluate` ~6 s, `add_predictions` ~5 s, `feature_importance` ~18 s,
`explain_prediction` ~8 s.

## `finetune_foundation_model(session_key, table, target, task, name=None, max_seconds=120)`

Runs gradient steps on the training split so the weights adapt to this one
table. It is usually unnecessary, because the pre-trained model already works.
Use it when asked to fine-tune, or when the pre-trained model's held-out score
is disappointing.

**Guardrails:**

| Guardrail | Behaviour |
|---|---|
| Installed | If the foundation model isn't installed, the tool declines and points at `install_foundation_model`. |
| Row cap | Above **5,000 rows**, the tool declines (too slow on a CPU). Use `train_*` with `backend="tabicl"` instead. |
| Time budget | Hard limit, default 120 s, capped at 600 s. Training runs at most 10 epochs, with early stopping on an internal validation slice. |
| Comparison | The pre-trained model is scored on the **same** held-out split. Both numbers are reported in `metadata.backend_selection.finetune`. If fine-tuning didn't improve the held-out score, a caveat recommends the pre-trained model. |
| Time-limit caveat | If the run stops at its time limit, a caveat says that the number of epochs (and so the weights) depends on machine speed. |

The saved model (`backend: "tabicl_finetuned"`) works with every downstream tool.
Unlike an in-context model, which pickles to about 200 KB because its weights
reload from the cache, a fine-tuned model stores its own weights: about 110 MB
per saved model in the session directory.

## Determinism

TabICL runs on the CPU by default with `random_state=0`. The checkpoint files
are pinned (`tabicl-classifier-v2-20260212.ckpt`,
`tabicl-regressor-v2-20260212.ckpt`), so a `tabicl` upgrade can't silently
change which model answered. Training the same table twice gives identical
predictions, and this is tested.

To opt into a GPU, set `MEELU_FOUNDATION_DEVICE=cuda`. It is faster, but results
can differ in the last digits.

## Why TabICL (as of October 2026)

| Model | Package | Code / weights license | Commercial use | Notes |
|---|---|---|---|---|
| **TabICL v2** | `tabicl` 2.2.0 | BSD-3 / BSD-3 | **Yes** | Ungated HF download. Classification and regression. Light dependencies. Fine-tuning API. |
| TabPFN 3.5 / 3 / 2.6 / 2.5 | `tabpfn` 9.1.0 | Apache-2.0 / *non-commercial* | **No** | 9.x defaults to v3.5. Downloads need a Prior Labs token and a browser license flow, which can't complete inside a stdio MCP server. |
| TabPFN v2 | `tabpfn` (pinned `ModelVersion.V2`) | Apache-2.0 / Prior Labs License 1.1 | Yes, with attribution | Requires a "Built with PriorLabs-TabPFN" notice. On CPU it refuses more than 1,000 rows by default. Heavy dependencies (lightgbm, skrub, matplotlib). |
| `tabpfn-client` | `tabpfn-client` 0.6.1 | Apache-2.0 client | Paid API terms | Sends your data to Prior Labs' servers. That is a non-starter for a local-only engine. |
| Mitra | `autogluon.tabular[mitra]` 1.6.3 | Apache-2.0 / Apache-2.0 | Yes | Pulls in all of AutoGluon and pins `torch<2.14`. |
| TabDPT 1.3 | `tabdpt` 1.3.1 | Apache-2.0 / Apache-2.0 | Yes | Credible alternative. Needs `faiss-cpu` and `torch>=2.6`. |
| LimiX, SAP-RPT-1-OSS | git only | Apache-2.0 code / unclear weights | Unclear | Not on PyPI. The weight terms restrict commercial or non-research use. |

**Sources:**

- `tabicl`: [PyPI](https://pypi.org/project/tabicl/),
  [HF card](https://huggingface.co/jingang/TabICL)
- TabPFN weight licenses:
  [v2 (Prior Labs License 1.1)](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt),
  [v3.5 (non-commercial)](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE),
  [v2.5 (non-commercial)](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE),
  [README license section](https://github.com/PriorLabs/TabPFN#license)
- [`tabpfn-client`](https://github.com/PriorLabs/tabpfn-client)
- Mitra: [model card](https://huggingface.co/autogluon/mitra-classifier-2)
- [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference)
- [LimiX license](https://huggingface.co/stable-ai/LimiX-16M/blob/main/LICENSE.txt)
- [SAP-RPT-1-OSS](https://github.com/SAP-samples/sap-rpt-1-oss)

## Limits

| Limit | Detail |
|---|---|
| CPU latency | Every TabICL predict call re-reads the training context, so even a one-row prediction costs a few seconds on a CPU. Tables of hundreds to a few thousand rows are comfortable. Near the 10,000-row `auto` ceiling, expect tens of seconds per call. |
| Not causal | Like any model, it learns associations. Importance and SHAP describe the model, not the world. |
| One held-out split | The trust block reflects a single split. A different split or fresh data can score differently. |
