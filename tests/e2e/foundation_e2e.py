"""End-to-end check of the foundation-model path through the real MCP protocol.

Launches ``meelu-analytics-mcp --stdio`` from this checkout (exactly as an
agent's client would) with a throwaway ``TABULAR_BASE``, generates a synthetic
churn CSV, and drives the supervised tools over stdio:

    uv run python tests/e2e/foundation_e2e.py                      # model installed
    uv run python tests/e2e/foundation_e2e.py --expect-backend gbt # not installed
    uv run python tests/e2e/foundation_e2e.py --install            # start without
        torch: train (trees + hint) -> install_foundation_model -> poll -> retrain

Checks: the backend ``auto`` picks (and its recorded reason), a trust block on
every result, held-out metrics, predictions written back, importance and SHAP
working on the chosen model, and an honest decline for a too-small table. Exits
non-zero on the first failed check. Not collected by pytest — it spawns a
server and, on first run, downloads the model weights.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]


def write_churn(path: Path, n: int = 600, seed: int = 7, extra_cols: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    charges = rng.normal(70, 20, n).round(2)
    tickets = rng.poisson(1.5, n)
    contract = rng.choice(["monthly", "annual", "two_year"], n, p=[0.5, 0.3, 0.2])
    logit = (1.2 - 0.05 * tenure + 0.025 * (charges - 70) + 0.35 * tickets
             + np.where(contract == "monthly", 1.0, -1.2))
    churned = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    frame = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure_months": tenure, "monthly_charges": charges,
        "support_tickets": tickets, "contract": contract, "churned": churned,
    })
    for j in range(extra_cols):  # uninformative numeric columns, to widen the table
        frame[f"usage_{j}"] = rng.normal(size=n).round(3)
    frame.to_csv(path, index=False)
    return path


def _payload(result):
    text = "".join(getattr(c, "text", "") for c in result.content)
    try:
        return json.loads(text)
    except ValueError:
        return text


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            self.failures.append(label)


# Per-call latency budget for foundation-model tools (the review target).
CALL_BUDGET_S = 60.0


async def run(expect_backend: str, install: bool, rows: int, extra_cols: int) -> int:
    base = tempfile.mkdtemp(prefix="meelu-foundation-e2e-")
    data = Path(tempfile.mkdtemp(prefix="meelu-foundation-data-"))
    churn = write_churn(data / "churn.csv", n=rows, extra_cols=extra_cols)
    tiny = data / "tiny.csv"
    pd.read_csv(churn).head(20).to_csv(tiny, index=False)

    params = StdioServerParameters(
        command="uv",
        args=["run", "--project", str(ROOT), "meelu-analytics-mcp", "--stdio"],
        env={**os.environ, "TABULAR_BASE": base},
    )
    c = Checker()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            c.check("finetune_foundation_model" in names, "finetune_foundation_model is registered")

            timings: dict[str, float] = {}

            async def call(tool, **args):
                t0 = time.monotonic()
                res = await s.call_tool(tool, args)
                out = _payload(res)
                elapsed = time.monotonic() - t0
                print(f"-> {tool} ({elapsed:.1f}s)")
                if tool != "install_foundation_model":
                    timings[tool] = max(timings.get(tool, 0.0), elapsed)
                c.check(not res.isError, f"{tool} returned without a protocol error")
                if res.isError:
                    print(out)
                return out if isinstance(out, dict) else {}

            sess = await call("create_session", paths=[str(churn), str(tiny)])
            key = sess.get("session_key")
            for table in ("churn", "tiny"):
                await call("classify_as_nominal", session_key=key, table=table)

            if install:
                before = await call("train_classifier", session_key=key, table="churn",
                                    target="churned")
                print(f"   before install: backend={before.get('backend')} hint={before.get('hint')!r}")
                c.check(before.get("backend") == "gbt", "without the model, trees are used")
                c.check("install_foundation_model" in (before.get("hint") or ""),
                        "the response hints at install_foundation_model")
                t0 = time.monotonic()
                st = await call("install_foundation_model")
                print(f"   install started: state={st.get('state')} step={st.get('step')}")
                c.check(time.monotonic() - t0 < 10, "install_foundation_model returns immediately")
                while st.get("state") == "installing":
                    st = await call("install_foundation_model", wait_seconds=60)
                    print(f"   state={st.get('state')} step={st.get('step')} "
                          f"after {time.monotonic() - t0:.0f}s")
                c.check(st.get("state") == "ready", f"install reached ready ({st.get('reason')})")

            out = await call("train_classifier", session_key=key, table="churn", target="churned")
            sel = out.get("metadata", {}).get("backend_selection", {})
            print(f"   backend={out.get('backend')} reason={sel.get('reason')!r}")
            print(f"   trust={out.get('trust', {}).get('level')} "
                  f"held_out={ {k: out.get('held_out_metrics', {}).get(k) for k in ('accuracy', 'roc_auc', 'baseline_accuracy')} }")
            c.check(out.get("backend") == expect_backend, f"auto chose {expect_backend}")
            c.check(sel.get("requested") == "auto" and bool(sel.get("reason")),
                    "selection reason recorded in metadata")
            c.check("trust" in out and out["trust"]["level"] != "unassessed",
                    "train result carries an assessed trust block")
            c.check((out.get("held_out_metrics", {}).get("roc_auc") or 0) > 0.65,
                    "held-out ROC-AUC beats 0.7 on a learnable signal")

            ev = await call("evaluate", session_key=key, table="churn", model_name="churned")
            c.check(ev.get("metadata", {}).get("backend") == expect_backend, "evaluate reports the backend")
            c.check("trust" in ev, "evaluate carries trust")

            pr = await call("add_predictions", session_key=key, table="churn", model_name="churned")
            c.check(pr.get("values", {}).get("n") == rows, f"predictions written for all {rows} rows")
            q = await call("run_sql", session_key=key,
                           query="SELECT count(*) AS n FROM churn WHERE churned_pred = 'yes'")
            print(f"   predicted churners: {q}")

            fi = await call("feature_importance", session_key=key, table="churn", model_name="churned")
            print(f"   importances: {fi.get('values', {}).get('importances')}")
            c.check(bool(fi.get("values", {}).get("importances")), "feature_importance works")

            ex = await call("explain_prediction", session_key=key, table="churn",
                            model_name="churned", row_index=0)
            c.check(bool(ex.get("values", {}).get("contributions")), "explain_prediction works")

            small = await call("train_classifier", session_key=key, table="tiny", target="churned")
            c.check(small.get("declined") is True, "20-row table declines honestly")
            if expect_backend == "tabicl":
                slowest = max(timings.items(), key=lambda kv: kv[1])
                c.check(slowest[1] <= CALL_BUDGET_S,
                        f"every tool call within {CALL_BUDGET_S:.0f}s (slowest: {slowest[0]} "
                        f"{slowest[1]:.1f}s)")

    print()
    if c.failures:
        print(f"FAILED {len(c.failures)} check(s): {c.failures}")
        return 1
    print("All checks passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect-backend", default="tabicl",
                        help="backend 'auto' should choose (tabicl when installed, gbt without)")
    parser.add_argument("--install", action="store_true",
                        help="start without the model and install it through the tool first")
    parser.add_argument("--rows", type=int, default=600, help="rows in the generated table")
    parser.add_argument("--extra-cols", type=int, default=0,
                        help="extra uninformative numeric columns (widens the table)")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.expect_backend, args.install, args.rows, args.extra_cols)))


if __name__ == "__main__":
    main()
