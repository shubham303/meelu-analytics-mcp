"""The on-demand runtime, for real: fresh interpreters, a real install, no
foundation packages in the interpreter's own environment.

Each step runs in a NEW Python process from a throwaway venv that has only the
core server installed — so tabicl/torch can come only from the
``<TABULAR_BASE>/.deps/foundation`` folder that ``install_foundation_model``
builds. Covers: the real install + ``sys.path`` wiring, training, reloading the
session in another process, and a version-stamp mismatch.

Slow (a real ``uv pip install`` of PyTorch; the weights come from the Hugging
Face cache if already present) and needs ``uv`` plus network or a warm cache,
so it runs only when asked:

    MEELU_FOUNDATION_REAL_INSTALL=1 uv run pytest tests/test_foundation_runtime.py
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MEELU_FOUNDATION_REAL_INSTALL") != "1" or shutil.which("uv") is None,
    reason="set MEELU_FOUNDATION_REAL_INSTALL=1 (and have uv) to run the real-install tests",
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    """A core-only venv plus a data root whose runtime was installed for real."""
    work = tmp_path_factory.mktemp("runtime")
    venv = work / "venv"
    subprocess.run(["uv", "venv", str(venv), "--python", sys.executable, "-q"], check=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), "-e", str(ROOT)],
                   check=True)
    base = work / "base"
    base.mkdir()
    from test_foundation import _churn_csv
    csv = _churn_csv(work / "churn.csv")

    def run(code: str, timeout: int = 1800) -> dict:
        res = subprocess.run(
            [str(python), "-c", textwrap.dedent(code)], capture_output=True, text=True,
            timeout=timeout, env={**os.environ, "TABULAR_BASE": str(base)},
        )
        assert res.returncode == 0, res.stderr[-3000:]
        return json.loads(res.stdout.strip().splitlines()[-1])

    first = run("""
        import importlib.util, json
        from tabint.analysis.service.algorithms import foundation_install as fi
        print(json.dumps({"tabicl_in_core": importlib.util.find_spec("tabicl") is not None,
                          "state": fi.status()["state"]}))
    """)
    assert first == {"tabicl_in_core": False, "state": "not_installed"}

    installed = run("""
        import json
        from tabint.analysis.service.algorithms import foundation_install as fi
        fi.start(); fi.wait(1800)
        print(json.dumps(fi.status()))
    """)
    assert installed["state"] == "ready", installed
    return run, base, csv


_TRAIN = """
    import json, os, sys
    import tabint.analysis.tools as t
    tool = lambda n: t.mcp._tool_manager._tools[n].fn
    deps = os.path.join(os.path.realpath("{base}"), ".deps", "foundation")
    key = tool("create_session")(paths=["{csv}"])["session_key"]
    tool("classify_as_nominal")(session_key=key, table="churn")
    wired_before = deps in sys.path
    out = tool("train_classifier")(session_key=key, table="churn", target="churned")
    import tabicl
    print(json.dumps({{"key": key, "backend": out["backend"], "acc": out["held_out_metrics"]["accuracy"],
                      "wired_before": wired_before, "tabicl_from": tabicl.__file__}}))
"""


def test_real_install_trains_from_the_runtime_folder(runtime):
    run, base, csv = runtime
    out = run(_TRAIN.format(base=base, csv=csv))
    assert out["backend"] == "tabicl"
    assert out["wired_before"] is False  # only training put the folder on sys.path
    assert out["tabicl_from"].startswith(str(base.resolve() / ".deps" / "foundation"))


def test_a_new_process_reloads_the_saved_model(runtime):
    run, base, csv = runtime
    trained = run(_TRAIN.format(base=base, csv=csv))
    reloaded = run(f"""
        import json
        import tabint.analysis.tools as t
        tool = lambda n: t.mcp._tool_manager._tools[n].fn
        info = tool("session_info")(session_key="{trained['key']}")
        ev = tool("evaluate")(session_key="{trained['key']}", table="churn", model_name="churned")
        print(json.dumps({{"unloaded": info.get("unloaded_models"), "backend": ev["metadata"]["backend"],
                          "acc": ev["values"]["accuracy"]}}))
    """)
    assert reloaded["unloaded"] is None
    assert reloaded["backend"] == "tabicl" and reloaded["acc"] == trained["acc"]


def test_a_stamp_mismatch_is_not_installed_and_saved_models_report_why(runtime):
    run, base, csv = runtime
    trained = run(_TRAIN.format(base=base, csv=csv))
    stamp_file = base.resolve() / ".deps" / "foundation" / ".complete"
    original = stamp_file.read_text()
    try:
        stamp_file.write_text(json.dumps({**json.loads(original), "python": "cpython-299"}))
        out = run(f"""
            import json, sys
            import tabint.analysis.tools as t
            from tabint.analysis.service.algorithms import foundation_install as fi
            tool = lambda n: t.mcp._tool_manager._tools[n].fn
            st = fi.status()
            info = tool("session_info")(session_key="{trained['key']}")
            retrain = tool("train_classifier")(session_key="{trained['key']}", table="churn",
                                               target="churned", name="again")
            print(json.dumps({{"state": st["state"], "reason": st["reason"],
                              "unloaded": info.get("unloaded_models"),
                              "backend": retrain["backend"], "hint": retrain.get("hint"),
                              "wired": any(".deps" in p for p in sys.path)}}))
        """)
        assert out["state"] == "not_installed" and "stale" in out["reason"]
        assert "install_foundation_model" in out["unloaded"]["churn"]["churned"]
        assert out["backend"] == "gbt" and "install_foundation_model" in out["hint"]
        assert out["wired"] is False
    finally:
        stamp_file.write_text(original)
