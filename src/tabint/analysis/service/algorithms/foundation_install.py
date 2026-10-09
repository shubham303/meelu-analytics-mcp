"""On-demand runtime for the foundation model — installed only when asked for.

PyTorch + TabICL are several hundred MB. The server's own install must stay fast,
so they are never part of it: an agent calls ``install_foundation_model``, which
starts :func:`start` — a background thread that

1. ``uv pip install --target <data root>/.deps/foundation`` the pinned packages
   (CPU-only torch; versions constrained to the running numpy / scikit-learn /
   scipy / pandas so nothing the core engine imports can be shadowed), then
2. downloads the two pinned checkpoints into the Hugging Face cache,

and returns at once; repeat calls report ``installing`` / ``ready`` / ``failed``.

The folder lives under the data root, outside the package's own environment, so
it survives ``uv tool install --force`` upgrades and ``uvx`` cache wipes. It is
appended to ``sys.path`` lazily (:func:`wire`) — only on the foundation code
path, and *after* the core environment, so the core's own numpy/sklearn always
win. A failed or interrupted install never leaves a half-usable folder: packages
land in a temporary sibling that is renamed into place only on success.

"Installed" means packages importable AND weights cached, so a training call
never starts a multi-hundred-MB download itself.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Pinned checkpoints: the model a result was produced with must not change under
# the user when tabicl ships a new default. Bumping these is a deliberate change.
CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
WEIGHTS_REPO = "jingang/TabICL"  # where tabicl itself fetches these checkpoints

# What gets installed. transformers is only needed by tabicl's fine-tuning
# scheduler; tabicl[finetune] would also drag in wandb, so it is named directly.
PACKAGES = ("tabicl>=2.2,<2.3", "transformers>=4.46,<6", "torch>=2.2")
# Core libraries the engine already imports: the install is constrained to the
# exact running versions so the two environments agree.
_CORE_PINS = ("numpy", "scipy", "scikit-learn", "pandas")
_INSTALL_TIMEOUT = 45 * 60
_COMPLETE_MARKER = ".complete"

_lock = threading.Lock()
_state: dict[str, Any] = {"state": None, "step": None, "reason": None,
                          "started_at": None, "finished_at": None}
_thread: threading.Thread | None = None


def deps_dir() -> Path:
    """``<data root>/.deps/foundation`` — the persistent runtime folder."""
    from ....shared import server  # lazy: shared.server imports this package
    from ..workspace import data_root
    return data_root(server._BASE) / ".deps" / "foundation"


def wire() -> bool:
    """Append the runtime folder to ``sys.path`` if a completed install exists.

    Appended, never prepended: anything the core environment already provides is
    found there first.
    """
    target = deps_dir()
    if not (target / _COMPLETE_MARKER).exists():
        return False
    if str(target) not in sys.path:
        sys.path.append(str(target))
        importlib.invalidate_caches()
    return True


def packages_installed() -> bool:
    """tabicl + torch importable (from the core env or the runtime folder)."""
    wire()
    return all(importlib.util.find_spec(m) is not None for m in ("tabicl", "torch"))


def weights_cached() -> bool:
    """Both pinned checkpoints already in the local Hugging Face cache. Checks
    the cache only — never touches the network."""
    if importlib.util.find_spec("huggingface_hub") is None:
        return False
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(WEIGHTS_REPO, f), str)
               for f in (CLASSIFIER_CHECKPOINT, REGRESSOR_CHECKPOINT))


def installed() -> bool:
    return packages_installed() and weights_cached()


def status() -> dict[str, Any]:
    """Where the install stands, plus what is on disk right now."""
    with _lock:
        out = dict(_state)
    packages, weights = packages_installed(), weights_cached()
    if out["state"] not in ("installing", "failed"):
        out["state"] = "ready" if (packages and weights) else "not_installed"
    out.update(packages_installed=packages, weights_cached=weights,
               deps_dir=str(deps_dir()))
    return out


def start() -> dict[str, Any]:
    """Begin the background install unless it is running or already done.
    A failed install is retried by calling this again."""
    global _thread
    if installed():
        return status()
    with _lock:
        if _state["state"] == "installing":
            return {**_state}
        _state.update(state="installing", step="starting", reason=None,
                      started_at=time.time(), finished_at=None)
        _thread = threading.Thread(target=_run, name="foundation-install", daemon=True)
        _thread.start()
    return status()


def wait(timeout: float) -> None:
    """Block up to ``timeout`` seconds for a running install to finish."""
    thread = _thread
    if thread is not None and timeout > 0:
        thread.join(timeout)


def _set(**kw: Any) -> None:
    with _lock:
        _state.update(**kw)


def _run() -> None:
    try:
        if not packages_installed():
            _set(step="installing packages (PyTorch CPU, TabICL)")
            _install_packages()
        if not weights_cached():
            _set(step="downloading model weights (~220 MB)")
            _prefetch_weights()
        if not installed():
            raise RuntimeError("install finished but tabicl/torch or the weights are "
                               "still not available")
        _set(state="ready", step=None, finished_at=time.time())
    except Exception as exc:  # reported through status(), never raised into a tool
        _set(state="failed", step=None, reason=f"{type(exc).__name__}: {exc}",
             finished_at=time.time())


def _uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for d in (Path.home() / ".local/bin", Path.home() / ".cargo/bin"):
        cand = d / ("uv.exe" if os.name == "nt" else "uv")
        if cand.exists():
            return str(cand)
    return None


def _constraints(path: Path) -> Path:
    lines = []
    for name in _CORE_PINS:
        try:
            lines.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    path.write_text("\n".join(lines) + "\n")
    return path


def install_command(target: Path, constraints: Path) -> list[str]:
    """The installer invocation: uv when available, else pip; CPU torch on Linux
    (the default Linux wheel bundles several GB of CUDA libraries)."""
    linux = platform.system() == "Linux"
    uv = _uv()
    if uv:
        cmd = [uv, "pip", "install", "--target", str(target), "--python", sys.executable,
               "--constraint", str(constraints), *PACKAGES]
        if linux:
            cmd += ["--torch-backend", "cpu"]
        return cmd
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
           "--target", str(target), "--constraint", str(constraints), *PACKAGES]
    if linux:
        cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    return cmd


def _install_packages() -> None:
    final = deps_dir()
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="foundation.tmp-", dir=final.parent))
    try:
        cmd = install_command(staging / "site", _constraints(staging / "constraints.txt"))
        env = {**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "120")}
        # stdin/stdout must never be inherited: on stdio they ARE the protocol.
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             timeout=_INSTALL_TIMEOUT, env=env, check=False)
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip()[-600:]
            raise RuntimeError(f"package install failed ({cmd[0]} exit {res.returncode}): {tail}")
        site = staging / "site"
        site.mkdir(exist_ok=True)
        (site / _COMPLETE_MARKER).write_text(" ".join(PACKAGES) + "\n")
        if final.exists():
            shutil.rmtree(final)
        site.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    wire()


_PREFETCH = """
import sys
from huggingface_hub import hf_hub_download
for f in sys.argv[2:]:
    hf_hub_download(repo_id=sys.argv[1], filename=f)
"""


def _prefetch_weights() -> None:
    """Download the checkpoints in a child process (keeps the server's own
    stdout and import state untouched)."""
    path = os.pathsep.join(p for p in (os.environ.get("PYTHONPATH"), str(deps_dir())) if p)
    res = subprocess.run(
        [sys.executable, "-c", _PREFETCH, WEIGHTS_REPO, CLASSIFIER_CHECKPOINT, REGRESSOR_CHECKPOINT],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=_INSTALL_TIMEOUT,
        env={**os.environ, "PYTHONPATH": path}, check=False,
    )
    if res.returncode != 0:
        tail = (res.stderr or "").strip()[-600:]
        raise RuntimeError(f"weight download failed: {tail}")
