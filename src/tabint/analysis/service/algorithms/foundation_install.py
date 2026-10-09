"""On-demand runtime for the foundation model — installed only when asked for.

PyTorch + TabICL are several hundred MB. The server's own install must stay fast,
so they are never part of it: an agent calls ``install_foundation_model``, which
starts :func:`start` — a background thread that

1. ``uv pip install --target <data root>/.deps/foundation`` the pinned packages
   (CPU-only torch; every distribution already in the core environment is
   pinned to its running version, so the two environments agree), then
2. downloads the two pinned checkpoints into the Hugging Face cache,

and returns at once; repeat calls report ``installing`` / ``ready`` / ``failed``.

The folder lives under the data root, outside the package's own environment, so
it survives ``uv tool install --force`` upgrades and ``uvx`` cache wipes. Its
``.complete`` stamp records the interpreter, platform and core pins it was built
for; after a Python or core-library upgrade the stamp no longer matches, the
folder counts as not installed, and the next install replaces it.

Several server processes may share one data root, so the package step runs
under an inter-process file lock, sweeps staging folders left by a killed
install, and swaps the finished folder in with ``os.replace`` — a complete,
matching folder is never removed.

The folder joins ``sys.path`` (appended, so the core environment always wins)
only through :func:`wire`, which is called only when a TabICL model is actually
built or unpickled. Checking whether the model is installed never wires it —
otherwise the folder's extra packages (fsspec, jinja2, sympy, …) would start
satisfying the core libraries' optional imports.

"Installed" means packages present AND weights cached, so a training call never
starts a multi-hundred-MB download itself.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator

# Pinned checkpoints: the model a result was produced with must not change under
# the user when tabicl ships a new default. Bumping these is a deliberate change.
CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
WEIGHTS_REPO = "jingang/TabICL"  # where tabicl itself fetches these checkpoints

# What gets installed. transformers is only needed by tabicl's fine-tuning
# scheduler; tabicl[finetune] would also drag in wandb, so it is named directly.
PACKAGES = ("tabicl>=2.2,<2.3", "transformers>=4.46,<6", "torch>=2.2")
_RUNTIME_MODULES = ("tabicl", "torch")
# Never pinned: the server's own distribution, and packaging tools.
_UNPINNED = {"meelu-analytics-mcp", "pip", "setuptools", "wheel", "uv"}
_INSTALL_TIMEOUT = 45 * 60
_STAMP = ".complete"
_LOCK = ".install.lock"

_lock = threading.Lock()
_state: dict[str, Any] = {"state": None, "step": None, "reason": None,
                          "started_at": None, "finished_at": None}
_thread: threading.Thread | None = None


# --------------------------------------------------------------------------- #
# where it lives, and whether it matches this interpreter
# --------------------------------------------------------------------------- #

def deps_root() -> Path:
    from ....shared import server  # lazy: shared.server imports this package
    from ..workspace import data_root
    return data_root(server._BASE) / ".deps"


def deps_dir() -> Path:
    """``<data root>/.deps/foundation`` — the persistent runtime folder."""
    return deps_root() / "foundation"


def _core_pins() -> dict[str, str]:
    """name → version for every distribution in the running core environment
    (excluding the runtime folder itself, which is never on sys.path here
    unless wired — and wired distributions are filtered by location)."""
    target = str(deps_dir())
    pins: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if not name or name in _UNPINNED:
            continue
        location = str(getattr(dist, "_path", ""))
        if location.startswith(target):
            continue
        pins[name] = dist.version
    return pins


def expected_stamp() -> dict[str, Any]:
    """What a runtime folder must have been built for to be usable here."""
    core = _core_pins()
    return {
        "python": sys.implementation.cache_tag,  # e.g. cpython-312
        "platform": sysconfig.get_platform(),     # e.g. macosx-11.0-arm64
        "packages": list(PACKAGES),
        "core": {k: core[k] for k in sorted(core)
                 if k in ("numpy", "scipy", "scikit-learn", "pandas")},
    }


def _read_stamp(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads((path / _STAMP).read_text())
    except (OSError, ValueError):
        return None


def stamp_problem(path: Path | None = None) -> str | None:
    """Why the runtime folder at ``path`` can't be used here, or None if it can."""
    path = path or deps_dir()
    stamp = _read_stamp(path)
    if stamp is None:
        return "no completed install"
    expected = expected_stamp()
    for key in ("python", "platform", "packages", "core"):
        if stamp.get(key) != expected[key]:
            return (f"built for a different {key} ({stamp.get(key)!r}; this server needs "
                    f"{expected[key]!r}) — it will be reinstalled")
    return None


def _folder_has_runtime(path: Path) -> bool:
    return all((path / m).is_dir() for m in _RUNTIME_MODULES)


def _core_has_runtime() -> bool:
    """tabicl + torch importable from the core environment itself (the optional
    ``foundation`` extra). find_spec on top-level names never imports torch."""
    target = str(deps_dir())
    for m in _RUNTIME_MODULES:
        spec = importlib.util.find_spec(m)
        if spec is None or (spec.origin or "").startswith(target):
            return False
    return True


def packages_installed() -> bool:
    """tabicl + torch available — from the core env, or from a runtime folder
    whose stamp matches this interpreter. Never touches sys.path."""
    if _core_has_runtime():
        return True
    return stamp_problem() is None and _folder_has_runtime(deps_dir())


def _hf_hub_cache() -> Path:
    """The Hugging Face hub cache directory, resolved like huggingface_hub does,
    without importing it (it may live only in the runtime folder)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "huggingface" / "hub"


def weights_cached() -> bool:
    """Both pinned checkpoints resolvable offline — at the snapshot ``main``
    points to, which is exactly where tabicl looks with downloads disabled."""
    repo = _hf_hub_cache() / ("models--" + WEIGHTS_REPO.replace("/", "--"))
    try:
        revision = (repo / "refs" / "main").read_text().strip()
    except OSError:
        return False
    snapshot = repo / "snapshots" / revision
    return all((snapshot / f).exists() for f in (CLASSIFIER_CHECKPOINT, REGRESSOR_CHECKPOINT))


def installed() -> bool:
    return packages_installed() and weights_cached()


def wire() -> bool:
    """Put a matching runtime folder on ``sys.path`` (appended — anything the
    core environment provides is still found there first). Call this only when a
    TabICL model is about to be built or unpickled."""
    if _core_has_runtime():
        return True
    target = deps_dir()
    if stamp_problem(target) is not None:
        return False
    if str(target) not in sys.path:
        sys.path.append(str(target))
        importlib.invalidate_caches()
    return True


def runtime_versions() -> dict[str, str | None]:
    """tabicl / torch versions actually in use (call after wire())."""
    out: dict[str, str | None] = {}
    for name in _RUNTIME_MODULES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


# --------------------------------------------------------------------------- #
# the background install
# --------------------------------------------------------------------------- #

def status() -> dict[str, Any]:
    """Where the install stands, plus what is on disk right now. Packages and
    weights present means ready, whatever an earlier attempt reported."""
    with _lock:
        out = dict(_state)
    packages, weights = packages_installed(), weights_cached()
    if packages and weights:
        out.update(state="ready", step=None, reason=None)
    elif out["state"] not in ("installing", "failed"):
        out["state"] = "not_installed"
    problem = None if _core_has_runtime() else stamp_problem()
    if problem and problem != "no completed install" and out["state"] == "not_installed":
        out["reason"] = f"Existing runtime folder is stale: {problem}."
    out.update(packages_installed=packages, weights_cached=weights,
               deps_dir=str(deps_dir()))
    return out


def start() -> dict[str, Any]:
    """Begin the background install unless it is running or already done.
    A failed install is retried by calling this again."""
    global _thread
    if installed():
        with _lock:
            _state.update(state="ready", step=None, reason=None)
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
        _set(state="ready", step=None, reason=None, finished_at=time.time())
    except Exception as exc:  # reported through status(), never raised into a tool
        _set(state="failed", step=None, reason=f"{type(exc).__name__}: {exc}",
             finished_at=time.time())


@contextlib.contextmanager
def _interprocess_lock() -> Iterator[None]:
    """Exclusive lock on ``.deps/.install.lock`` shared by every server process
    using this data root. POSIX ``flock``; on Windows ``msvcrt.locking``."""
    root = deps_root()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / _LOCK, "a+") as fh:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt
            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _sweep(root: Path) -> None:
    """Remove staging/retired folders. Only called under the lock, when no other
    install can be using them — so anything left is from a killed install."""
    for stale in list(root.glob("foundation.tmp-*")) + list(root.glob("foundation.old-*")):
        shutil.rmtree(stale, ignore_errors=True)


def _uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for d in (Path(sys.executable).parent, Path.home() / ".local/bin", Path.home() / ".cargo/bin"):
        cand = d / ("uv.exe" if os.name == "nt" else "uv")
        if cand.exists():
            return str(cand)
    return None


def _write_constraints(path: Path) -> Path:
    pins = _core_pins()
    path.write_text("".join(f"{name}=={ver}\n" for name, ver in sorted(pins.items())))
    return path


def install_command(target: Path, constraints: Path) -> list[str]:
    """The installer invocation: uv (``--torch-backend cpu`` on Linux, whose
    default torch wheel bundles GBs of CUDA) when available; pip with the
    PyTorch CPU index only as a fallback."""
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
    root = deps_root()
    final = deps_dir()
    with _interprocess_lock():
        # Another process may have finished while we waited for the lock.
        if stamp_problem(final) is None and _folder_has_runtime(final):
            return
        _sweep(root)
        staging = Path(tempfile.mkdtemp(prefix="foundation.tmp-", dir=root))
        try:
            site = staging / "site"
            cmd = install_command(site, _write_constraints(staging / "constraints.txt"))
            env = {**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "120")}
            # stdin/stdout must never be inherited: on stdio they ARE the protocol.
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                 timeout=_INSTALL_TIMEOUT, env=env, check=False)
            if res.returncode != 0:
                tail = (res.stderr or res.stdout or "").strip()[-600:]
                raise RuntimeError(
                    f"package install failed ({Path(cmd[0]).name} exit {res.returncode}): {tail}")
            site.mkdir(exist_ok=True)
            (site / _STAMP).write_text(json.dumps(expected_stamp(), indent=1))
            # Atomic swap: retire the stale folder under a sibling name, then
            # move the new one into place. Only a stale/incomplete folder gets
            # here — a matching one returned above.
            if final.exists():
                os.replace(final, root / f"foundation.old-{os.getpid()}-{time.time_ns()}")
            os.replace(site, final)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            _sweep(root)


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
