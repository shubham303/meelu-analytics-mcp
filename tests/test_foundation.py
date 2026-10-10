"""The foundation-model backend: deterministic selection, honest declines, and a
working model downstream.

The first half runs everywhere — it fakes the extra being absent or the weights
failing to load. The second half needs the real `foundation` extra and skips
cleanly without it.
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabint.analysis.service.algorithms import foundation, foundation_install
from tabint.shared import server as shared_server


def _churn_csv(path: Path, n: int = 240, seed: int = 0) -> Path:
    """A small synthetic churn table with a learnable signal."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    charges = rng.normal(70, 20, n).round(2)
    contract = rng.choice(["monthly", "annual", "two_year"], n, p=[0.5, 0.3, 0.2])
    logit = 1.5 - 0.06 * tenure + 0.02 * (charges - 70) + np.where(contract == "monthly", 1.0, -1.0)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    pd.DataFrame({
        "tenure": tenure, "monthly_charges": charges,
        "contract": contract, "churned": churned,
    }).to_csv(path, index=False)
    return path


def _tool(name):
    import tabint.analysis.tools as tools
    return tools.mcp._tool_manager._tools[name].fn


@pytest.fixture()
def churn(tmp_path, monkeypatch):
    """A live session on the churn table, with the categorical column refined."""
    monkeypatch.setenv("TABULAR_BASE", str(tmp_path))
    monkeypatch.setattr(shared_server, "_BASE", str(tmp_path))
    csv = _churn_csv(tmp_path / "churn.csv")
    key = _tool("create_session")(paths=[str(csv)])["session_key"]
    _tool("classify_as_nominal")(session_key=key, table="churn")
    return key


def _train(key, **kw):
    return _tool("train_classifier")(session_key=key, table="churn", target="churned", **kw)


# --------------------------------------------------------------------------- #
# Without the extra / with the weights unavailable — runs everywhere
# --------------------------------------------------------------------------- #

class _Unloadable:
    """Stands in for a foundation estimator whose weights can't be fetched."""

    def fit(self, X, y):
        raise OSError("We couldn't connect to 'https://huggingface.co' (offline)")

    def get_params(self, deep=True):
        return {}


def test_not_installed_trains_trees_and_hints_how_to_install(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: False)
    out = _train(churn)
    assert out["backend"] == "gbt"
    sel = out["metadata"]["backend_selection"]
    assert sel["requested"] == "auto" and sel["chosen"] == "gbt"
    assert sel["checks"]["foundation_installed"] is False
    assert "not installed" in sel["reason"]
    assert "install_foundation_model" in out["hint"]
    assert out["trust"]["level"] in ("high", "moderate", "low")
    assert out["declined"] is False


def test_no_hint_when_the_table_is_outside_the_envelope(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: False)
    monkeypatch.setattr(foundation, "AUTO_MAX_ROWS", 100)
    out = _train(churn)
    assert out["backend"] == "gbt" and "hint" not in out


def test_auto_install_env_starts_the_background_install(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: False)
    monkeypatch.setenv("MEELU_FOUNDATION_AUTO_INSTALL", "1")
    started = []
    monkeypatch.setattr(foundation_install, "start",
                        lambda: started.append(1) or {"state": "installing"})
    out = _train(churn)
    assert started and out["backend"] == "gbt"
    assert "installing" in out["hint"]


def test_explicit_foundation_without_the_extra_declines(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: False)
    out = _train(churn, backend="tabicl")
    assert out["declined"] is True
    assert "install_foundation_model" in out["trust"]["decline_reason"]


def test_finetune_without_the_extra_declines(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: False)
    out = _tool("finetune_foundation_model")(
        session_key=churn, table="churn", target="churned", task="classification")
    assert out["declined"] is True


def test_unloadable_weights_fall_back_under_auto(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: True)
    monkeypatch.setattr(foundation, "make_estimator", lambda task: _Unloadable())
    out = _train(churn)
    assert out["backend"] == "gbt"
    sel = out["metadata"]["backend_selection"]
    assert "install_foundation_model" in sel["foundation_error"]
    assert any("foundation model was unavailable" in c for c in out["trust"]["caveats"])


def test_unloadable_weights_decline_when_requested_explicitly(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: True)
    monkeypatch.setattr(foundation, "make_estimator", lambda task: _Unloadable())
    out = _train(churn, backend="tabicl")
    assert out["declined"] is True
    assert "weights" in out["trust"]["decline_reason"]


from sklearn.base import BaseEstimator, ClassifierMixin


class _FailsAtPredict(ClassifierMixin, BaseEstimator):
    """Loads fine but its forward pass fails — TabICL's fit only loads weights."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        raise RuntimeError("CPU out of memory during forward pass")


def test_forward_pass_failure_falls_back_under_auto(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: True)
    monkeypatch.setattr(foundation, "make_estimator", lambda task: _FailsAtPredict())
    out = _train(churn)
    assert out["backend"] == "gbt"
    assert "out of memory" in out["metadata"]["backend_selection"]["foundation_error"]


def test_forward_pass_failure_declines_when_explicit(churn, monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: True)
    monkeypatch.setattr(foundation, "make_estimator", lambda task: _FailsAtPredict())
    out = _train(churn, backend="tabicl")
    assert out["declined"] is True


def test_python_api_default_is_still_trees(churn, monkeypatch):
    """Only the MCP tools default to "auto"; library callers keep "gbt"."""
    monkeypatch.setattr(foundation, "available", lambda: True)
    from tabint.shared.server import get_session
    model = get_session(churn).table("churn").train_classifier("churned", name="lib")
    assert model._backend == "gbt"


def test_batched_permutation_importance_matches_the_signal(tmp_path, monkeypatch):
    """The foundation-model importance path, exercised on a tree model."""
    from tabint.analysis.service.algorithms import supervised
    monkeypatch.setattr(shared_server, "_BASE", str(tmp_path))
    rng = np.random.default_rng(2)
    x1, x2 = rng.normal(size=400), rng.normal(size=400)
    csv = tmp_path / "sig.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "y": (x1 > 0).astype(int)}).to_csv(csv, index=False)
    key = _tool("create_session")(paths=[str(csv)])["session_key"]
    _tool("train_classifier")(session_key=key, table="sig", target="y", backend="gbt")
    monkeypatch.setattr(supervised.TrainedModel, "is_foundation", property(lambda self: True))
    fi = _tool("feature_importance")(session_key=key, table="sig", model_name="y")
    imp = fi["values"]["importances"]
    assert fi["metadata"]["n_repeats"] == 3
    assert imp["x1"] > 0.3 and abs(imp["x2"]) < 0.05


def test_auto_rule_is_a_pure_function_of_shape(monkeypatch):
    monkeypatch.setattr(foundation, "available", lambda: True)
    pick = lambda **kw: foundation.select_backend(
        "auto", task="classification", **{"n_rows": 500, "n_features": 10, "n_classes": 2, **kw}
    )["chosen"]
    assert pick() == "tabicl"
    assert pick(n_rows=foundation.AUTO_MAX_ROWS + 1) == "gbt"
    assert pick(n_rows=1_000, n_features=foundation.AUTO_MAX_CELLS // 1_000 + 1) == "gbt"
    assert pick(n_rows=foundation.AUTO_MAX_ROWS, n_features=foundation.AUTO_MAX_CELLS
                // foundation.AUTO_MAX_ROWS) == "tabicl"
    assert pick(n_classes=foundation.AUTO_MAX_CLASSES + 1) == "gbt"
    assert pick() == pick()


def test_no_skill_model_is_low_trust(tmp_path, monkeypatch):
    """A model that can't beat the majority class must not read as trustworthy."""
    monkeypatch.setattr(shared_server, "_BASE", str(tmp_path))
    monkeypatch.setattr(foundation, "available", lambda: False)
    rng = np.random.default_rng(1)
    csv = tmp_path / "noise.csv"
    pd.DataFrame({"x": rng.normal(size=400), "y": (rng.random(400) < 0.8).astype(int)}).to_csv(csv, index=False)
    key = _tool("create_session")(paths=[str(csv)])["session_key"]
    out = _tool("train_classifier")(session_key=key, table="noise", target="y")
    # Pure noise: trees can't beat always-predict-majority on this seed.
    assert out["held_out_metrics"]["accuracy"] <= out["held_out_metrics"]["baseline_accuracy"]
    assert out["trust"]["level"] == "low"
    assert any("most common class" in c and "ROC-AUC" in c for c in out["trust"]["caveats"])


# --------------------------------------------------------------------------- #
# On-demand install: status transitions (subprocess mocked) and sys.path wiring
# --------------------------------------------------------------------------- #

@pytest.fixture()
def fresh_install(tmp_path, monkeypatch):
    """A clean install state under a temp data root. The core venv's own tabicl is
    hidden (``_core_has_runtime`` → False) so only the runtime folder counts; the
    real stamp logic decides whether it is usable; weights are a flag."""
    monkeypatch.setattr(shared_server, "_BASE", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(foundation_install, "_state", {
        "state": None, "step": None, "reason": None, "started_at": None, "finished_at": None})
    monkeypatch.setattr(foundation_install, "_core_has_runtime", lambda: False)
    weights = {"cached": False}
    monkeypatch.setattr(foundation_install, "weights_cached", lambda: weights["cached"])
    return tmp_path, weights


def _completed(cmd, rc=0, err=""):
    import subprocess
    return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)


def _fake_installer(weights, calls, delay=0.0):
    """Stands in for uv/pip: lays down tabicl/ and torch/ in --target; the weight
    prefetch call just flips the flag."""
    def fake_run(cmd, **kw):
        calls.append(cmd)
        assert kw["stdin"] is not None and kw["capture_output"]  # never touch stdio
        if "--target" in cmd:
            time.sleep(delay)
            target = Path(cmd[cmd.index("--target") + 1])
            for pkg in ("tabicl", "torch"):
                (target / pkg).mkdir(parents=True)
        else:
            weights["cached"] = True
        return _completed(cmd)
    return fake_run


def _write_runtime(path: Path, stamp: dict) -> None:
    for pkg in ("tabicl", "torch"):
        (path / pkg).mkdir(parents=True, exist_ok=True)
    (path / ".complete").write_text(json.dumps(stamp))


def test_install_goes_installing_then_ready(fresh_install, monkeypatch):
    root, weights = fresh_install
    calls = []
    monkeypatch.setattr(foundation_install.subprocess, "run", _fake_installer(weights, calls))
    assert foundation_install.status()["state"] == "not_installed"
    first = foundation_install.start()
    assert first["state"] in ("installing", "ready")
    foundation_install.wait(10)
    final = foundation_install.status()
    assert final["state"] == "ready", final
    stamp = json.loads((root / ".deps" / "foundation" / ".complete").read_text())
    assert stamp == foundation_install.expected_stamp()
    install_cmd = calls[0]
    assert "tabicl>=2.2,<2.3" in install_cmd and "--constraint" in install_cmd
    assert not list((root / ".deps").glob("foundation.tmp-*"))


def test_constraints_pin_every_core_distribution(tmp_path):
    pins = foundation_install._write_constraints(tmp_path / "c.txt").read_text().splitlines()
    names = {line.split("==")[0] for line in pins}
    assert {"numpy", "scikit-learn", "pandas", "scipy", "duckdb"} <= names
    assert "meelu-analytics-mcp" not in names


def test_failed_install_reports_the_reason_and_leaves_nothing_behind(fresh_install, monkeypatch):
    root, _ = fresh_install
    monkeypatch.setattr(foundation_install.subprocess, "run",
                        lambda cmd, **kw: _completed(cmd, 1, "No solution found for torch"))
    foundation_install.start()
    foundation_install.wait(10)
    st = foundation_install.status()
    assert st["state"] == "failed"
    assert "No solution found for torch" in st["reason"]
    assert not (root / ".deps" / "foundation").exists()
    assert not list((root / ".deps").glob("foundation.tmp-*"))


def test_failed_is_not_sticky_once_everything_is_present(fresh_install, monkeypatch):
    _, weights = fresh_install
    foundation_install._state.update(state="failed", reason="earlier network error")
    _write_runtime(foundation_install.deps_dir(), foundation_install.expected_stamp())
    weights["cached"] = True
    assert foundation_install.status()["state"] == "ready"
    assert foundation_install.start()["state"] == "ready"
    assert foundation_install._state["state"] == "ready"


def test_a_stale_runtime_counts_as_not_installed_and_is_replaced(fresh_install, monkeypatch):
    """After a Python upgrade the old cp312 torch must not count as installed."""
    root, weights = fresh_install
    weights["cached"] = True
    old = foundation_install.deps_dir()
    _write_runtime(old, {**foundation_install.expected_stamp(), "python": "cpython-299"})
    (old / "old_marker").write_text("")
    assert foundation_install.packages_installed() is False
    assert foundation_install.wire() is False and str(old) not in sys.path
    st = foundation_install.status()
    assert st["state"] == "not_installed" and "stale" in st["reason"]
    assert foundation.available() is False

    calls = []
    monkeypatch.setattr(foundation_install.subprocess, "run", _fake_installer(weights, calls))
    foundation_install.start()
    foundation_install.wait(10)
    assert foundation_install.status()["state"] == "ready"
    assert len(calls) == 1  # reinstalled
    assert not (old / "old_marker").exists()  # the stale folder was swapped out
    assert not list((root / ".deps").glob("foundation.old-*"))


def test_core_library_upgrade_also_invalidates_the_stamp(fresh_install, monkeypatch):
    stamp = foundation_install.expected_stamp()
    stamp["core"] = {**stamp["core"], "numpy": "0.0.1"}
    _write_runtime(foundation_install.deps_dir(), stamp)
    assert "core" in foundation_install.stamp_problem()


def test_install_sweeps_orphans_and_never_replaces_a_valid_runtime(fresh_install, monkeypatch):
    root, weights = fresh_install
    deps = root / ".deps"
    (deps / "foundation.tmp-killed").mkdir(parents=True)
    (deps / "foundation.old-killed").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(foundation_install.subprocess, "run", _fake_installer(weights, calls))
    foundation_install._install_packages()
    assert not list(deps.glob("foundation.tmp-*")) and not list(deps.glob("foundation.old-*"))
    # A complete, matching runtime is left alone: no reinstall, no removal.
    (foundation_install.deps_dir() / "keep").write_text("")
    foundation_install._install_packages()
    assert len(calls) == 1 and (foundation_install.deps_dir() / "keep").exists()


def test_concurrent_installs_are_serialised_by_the_file_lock(fresh_install, monkeypatch):
    """Two installers sharing one data root: the second waits on the lock, then
    finds the first one's runtime valid and does nothing."""
    _, weights = fresh_install
    calls = []
    monkeypatch.setattr(foundation_install.subprocess, "run",
                        _fake_installer(weights, calls, delay=0.5))
    errors = []

    def worker():
        try:
            foundation_install._install_packages()
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors and len(calls) == 1
    assert foundation_install.stamp_problem() is None


def test_checking_availability_never_touches_sys_path(fresh_install):
    _, weights = fresh_install
    weights["cached"] = True
    _write_runtime(foundation_install.deps_dir(), foundation_install.expected_stamp())
    before = list(sys.path)
    assert foundation.available() is True
    foundation.select_backend("auto", task="classification", n_rows=100, n_features=3, n_classes=2)
    foundation_install.status()
    assert sys.path == before


def test_install_tool_reports_status_and_next_step(fresh_install, monkeypatch):
    monkeypatch.setattr(foundation_install, "start", lambda: {"state": "installing"})
    monkeypatch.setattr(foundation_install, "status", lambda: {"state": "installing"})
    out = _tool("install_foundation_model")()
    assert out["state"] == "installing" and "again" in out["next_step"]


def test_wire_appends_the_runtime_folder_only_after_a_complete_install(fresh_install):
    target = foundation_install.deps_dir()
    target.mkdir(parents=True)
    (target / "meelu_wire_probe.py").write_text("VALUE = 42\n")
    assert foundation_install.wire() is False  # no stamp: half-installed is ignored
    assert str(target) not in sys.path
    (target / ".complete").write_text(json.dumps(foundation_install.expected_stamp()))
    assert foundation_install.wire() is True
    assert sys.path[-1] == str(target)  # appended: the core env still wins
    foundation_install.wire()
    assert sys.path.count(str(target)) == 1
    import importlib
    assert importlib.import_module("meelu_wire_probe").VALUE == 42
    sys.modules.pop("meelu_wire_probe", None)


def test_weights_cache_lookup_follows_refs_main(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--jingang--TabICL"
    assert foundation_install.weights_cached() is False
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / foundation_install.CLASSIFIER_CHECKPOINT).write_text("")
    assert foundation_install.weights_cached() is False
    (snap / foundation_install.REGRESSOR_CHECKPOINT).write_text("")
    assert foundation_install.weights_cached() is True


def test_install_command_uses_cpu_torch_on_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(foundation_install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(foundation_install, "_uv", lambda: "/usr/bin/uv")
    cmd = foundation_install.install_command(tmp_path / "t", tmp_path / "c.txt")
    assert cmd[:3] == ["/usr/bin/uv", "pip", "install"] and "--torch-backend" in cmd
    assert "--extra-index-url" not in cmd  # uv's torch backend is preferred
    monkeypatch.setattr(foundation_install, "_uv", lambda: None)
    cmd = foundation_install.install_command(tmp_path / "t", tmp_path / "c.txt")
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert "https://download.pytorch.org/whl/cpu" in cmd


def test_explainers_decline_above_the_scoring_budget(tmp_path, monkeypatch):
    """Too wide to explain within a call's CPU budget → an honest decline."""
    from tabint.analysis.service.algorithms import interpretation, supervised
    monkeypatch.setattr(shared_server, "_BASE", str(tmp_path))
    rng = np.random.default_rng(3)
    csv = tmp_path / "wide.csv"
    pd.DataFrame({**{f"x{i}": rng.normal(size=200) for i in range(6)},
                  "y": rng.integers(0, 2, 200)}).to_csv(csv, index=False)
    key = _tool("create_session")(paths=[str(csv)])["session_key"]
    _tool("train_classifier")(session_key=key, table="wide", target="y", backend="gbt")
    monkeypatch.setattr(supervised.TrainedModel, "is_foundation", property(lambda self: True))
    monkeypatch.setattr(interpretation, "_FOUNDATION_SCORED_ROWS", 40)
    fi = _tool("feature_importance")(session_key=key, table="wide", model_name="y")
    assert fi["declined"] is True and "CPU budget" in fi["trust"]["decline_reason"]
    ex = _tool("explain_prediction")(session_key=key, table="wide", model_name="y")
    assert ex["declined"] is True and "CPU budget" in ex["trust"]["decline_reason"]


# --------------------------------------------------------------------------- #
# With the real foundation extra
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tabicl():
    return pytest.importorskip("tabicl")


def test_auto_picks_the_foundation_model_and_downstream_tools_work(tabicl, churn):
    out = _train(churn)
    assert out["backend"] == "tabicl", out["metadata"]
    assert out["method"] == "classification:tabicl"
    assert out["metadata"]["backend_selection"]["chosen"] == "tabicl"
    assert out["held_out_metrics"]["roc_auc"] > 0.6
    assert out["trust"]["level"] in ("high", "moderate", "low")
    assert any("foundation model" in c for c in out["trust"]["caveats"])

    ev = _tool("evaluate")(session_key=churn, table="churn", model_name="churned")
    assert ev["metadata"]["backend"] == "tabicl"
    assert ev["values"]["accuracy"] == out["held_out_metrics"]["accuracy"]

    pr = _tool("add_predictions")(session_key=churn, table="churn", model_name="churned")
    assert pr["values"]["n"] == 240

    fi = _tool("feature_importance")(session_key=churn, table="churn", model_name="churned")
    assert set(fi["values"]["importances"]) == {"tenure", "monthly_charges", "contract"}
    assert fi["metadata"]["backend"] == "tabicl"

    ex = _tool("explain_prediction")(session_key=churn, table="churn", model_name="churned", row_index=3)
    assert ex["method"] == "shap_permutation"
    assert ex["metadata"]["explained_class"] == 1  # P(churn), as with trees
    assert set(ex["values"]["contributions"]) == {"tenure", "monthly_charges", "contract"}
    assert ex["trust"]["level"] == "moderate"


def test_foundation_model_never_writes_to_stdout(tabicl, churn, tmp_path, capsys):
    """On stdio, stdout is the protocol stream — the library's prints must not reach it."""
    from tabint.analysis.db import persistence
    capsys.readouterr()
    _train(churn)
    _tool("explain_prediction")(session_key=churn, table="churn", model_name="churned")
    persistence.open_session(churn, base=str(tmp_path))
    assert capsys.readouterr().out == ""


def test_foundation_training_is_deterministic(tabicl, churn):
    first = _train(churn, name="a")
    second = _train(churn, name="b")
    assert first["held_out_metrics"] == second["held_out_metrics"]
    assert first["metadata"]["backend_selection"] == second["metadata"]["backend_selection"]


def test_foundation_model_survives_a_session_reopen(tabicl, churn, tmp_path):
    from tabint.analysis.db import persistence
    before = _train(churn)["held_out_metrics"]
    reopened = persistence.open_session(churn, base=str(tmp_path))
    after = reopened.table("churn").evaluate("churned").values
    assert after["accuracy"] == before["accuracy"]


def test_regressor_uses_the_foundation_model(tabicl, churn):
    out = _tool("train_regressor")(session_key=churn, table="churn", target="monthly_charges")
    assert out["backend"] == "tabicl"
    assert "r2" in out["held_out_metrics"]


def test_finetune_reports_comparison_against_pretrained(tabicl, churn):
    pytest.importorskip("transformers")
    out = _tool("finetune_foundation_model")(
        session_key=churn, table="churn", target="churned", task="classification",
        name="tuned", max_seconds=20)
    assert out["backend"] == "tabicl_finetuned", out
    ft = out["metadata"]["backend_selection"]["finetune"]
    assert {"held_out_pretrained", "held_out_finetuned", "improved_over_pretrained"} <= set(ft)
    assert ft["time_limit_s"] == 20
    ev = _tool("evaluate")(session_key=churn, table="churn", model_name="tuned")
    assert ev["values"]["accuracy"] == pytest.approx(ft["held_out_finetuned"])


def test_finetune_declines_above_the_row_cap(tabicl, churn, monkeypatch):
    monkeypatch.setattr(foundation, "FINETUNE_MAX_ROWS", 100)
    out = _tool("finetune_foundation_model")(
        session_key=churn, table="churn", target="churned", task="classification")
    assert out["declined"] is True
    assert "too many to fine-tune" in out["trust"]["decline_reason"]


def test_selection_records_the_runtime(tabicl, churn):
    out = _train(churn)
    rt = out["metadata"]["backend_selection"]["foundation"]
    assert rt["device"] == "cpu" and rt["n_estimators"] == foundation.N_ESTIMATORS_CPU
    assert rt["checkpoints"]["classification"] == foundation_install.CLASSIFIER_CHECKPOINT
    assert rt["tabicl_version"] and rt["torch_version"]


def test_unloadable_saved_model_is_reported_then_recovers(tabicl, churn, tmp_path, monkeypatch):
    """Reopening where the runtime is missing: session_info names the model and
    the install hint; once it is back, the next use loads it."""
    from tabint.analysis.db import persistence
    from tabint.shared.server import _SESSIONS
    _train(churn)
    _SESSIONS.pop(churn, None)
    monkeypatch.setattr(foundation_install, "wire", lambda: False)
    info = _tool("session_info")(session_key=churn)
    reason = info["unloaded_models"]["churn"]["churned"]
    assert "install_foundation_model" in reason
    with pytest.raises(LookupError, match="install_foundation_model"):
        _tool("evaluate")(session_key=churn, table="churn", model_name="churned")
    monkeypatch.undo()
    monkeypatch.setattr(shared_server, "_BASE", str(tmp_path))
    ev = _tool("evaluate")(session_key=churn, table="churn", model_name="churned")
    assert ev["metadata"]["backend"] == "tabicl"


def test_add_predictions_declines_above_the_scoring_cap(tabicl, churn, monkeypatch):
    _train(churn)
    monkeypatch.setattr(foundation, "PREDICT_MAX_ROWS", 100)
    out = _tool("add_predictions")(session_key=churn, table="churn", model_name="churned")
    assert out["declined"] is True and "minutes" in out["trust"]["decline_reason"]
