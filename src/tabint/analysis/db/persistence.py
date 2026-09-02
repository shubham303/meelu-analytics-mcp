"""On-disk session persistence — the state layer under the CLI and MCP surfaces.

A *session* is an addressable, on-disk bundle identified by a short **session key**
(``s_ab12cd34``). It lets an external agent chain operations across separate
process invocations (CLI) or tool calls (MCP): create once, get the key back, then
reference it on every later call.

Layout::

    <base>/.tableint/sessions/<session_id>/
        data.duckdb     tables + write-back columns (persist for free via DuckDB)
        models/         pickled TrainedModel objects, <table>__<name>.pkl
        meta.json       session key, timestamps, tables, registered models

The DuckDB file makes tables and every materialized (write-back) column durable
automatically; only trained models need explicit pickling, since they live in the
Table.models registry rather than in the database.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session import Session
from ..service.workspace import Workspace

_SESSIONS_SUBDIR = Path(".tableint") / "sessions"

# nanoid-style key: short, URL/filename-safe, collision-resistant. Alphanumeric
# only (no -/_), so a key is safe as a directory name and a bare CLI argument.
_NANOID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NANOID_SIZE = 6


def _root(base: str | Path | None) -> Path:
    """Directory that holds all sessions, under the given base (default: cwd)."""
    return Path(base or ".").resolve() / _SESSIONS_SUBDIR


def session_dir(session_id: str, base: str | Path | None = None) -> Path:
    return _root(base) / session_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(name: str) -> str:
    """Filename-safe form of a table/model name for the models/ directory."""
    return re.sub(r"\W+", "_", name).strip("_") or "x"


def new_session_key() -> str:
    """Mint a fresh, opaque session key — an ``s_`` prefix plus a nanoid body."""
    return "s_" + "".join(secrets.choice(_NANOID_ALPHABET) for _ in range(_NANOID_SIZE))


def create_session(
    paths: list[str], base: str | Path | None = None, session_id: str | None = None
) -> Session:
    """Create a new persistent session from CSV paths; returns it with ``.id`` set.

    Args:
        paths: CSV files to load as the session's tables.
        base: Root under which ``.tableint/sessions`` lives (default: cwd).
        session_id: Optional explicit key; a fresh one is minted if omitted.

    Returns:
        A Session backed by an on-disk DuckDB, with ``.id`` = the session key.
    """
    session_id = session_id or new_session_key()
    sdir = session_dir(session_id, base)
    (sdir / "models").mkdir(parents=True, exist_ok=True)

    ws = Workspace.create(
        [str(p) for p in paths],
        db_path=str(sdir / "data.duckdb"),
        allowed_dir=base or ".",
    )
    session = Session._from_workspace(ws, session_id=session_id, session_dir=sdir)
    _write_meta(session)
    return session


def open_session(session_id: str, base: str | Path | None = None) -> Session:
    """Reattach to an existing session by key, restoring tables and models.

    Args:
        session_id: The session key returned by create_session.
        base: Root under which the session lives (default: cwd).

    Returns:
        A Session reattached to the on-disk DuckDB with its models reloaded.
    """
    sdir = session_dir(session_id, base)
    db = sdir / "data.duckdb"
    if not db.exists():
        raise FileNotFoundError(f"No session {session_id!r} at {sdir}")

    ws = Workspace(db_path=str(db), allowed_dir=base or ".")
    session = Session._from_workspace(ws, session_id=session_id, session_dir=sdir)
    _load_models(session)
    return session


def list_sessions(base: str | Path | None = None) -> list[str]:
    """List the keys of all persisted sessions under ``base``."""
    root = _root(base)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "data.duckdb").exists())


def save_model(session: Session, table_name: str, model_name: str, model: Any) -> None:
    """Persist a trained model into the session, keyed by table + model name.

    Called by the CLI/MCP after training so a later ``open_session`` can restore
    it into the table's model registry.
    """
    if session._dir is None:
        raise ValueError("save_model requires a persistent session.")
    path = session._dir / "models" / f"{_safe(table_name)}__{_safe(model_name)}.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"table": table_name, "name": model_name, "model": model}, fh)
    _write_meta(session)


def _load_models(session: Session) -> None:
    """Load every pickled model back into its table's registry."""
    mdir = session._dir / "models"
    if not mdir.exists():
        return
    for pkl in mdir.glob("*.pkl"):
        try:
            data = pickle.loads(pkl.read_bytes())
            session.workspace.table(data["table"]).models[data["name"]] = data["model"]
        except Exception:
            # A model whose table is gone or that fails to unpickle is skipped
            # rather than breaking the whole session open.
            continue


def _write_meta(session: Session) -> None:
    """Write/refresh meta.json, preserving created_at, tolerant of a corrupt file.

    Reads defensively (a truncated/corrupt meta must not poison later writes) and
    writes atomically (temp file + os.replace) so an interrupted write can never
    leave a half-written meta.json behind.
    """
    path = session._dir / "meta.json"
    try:
        existing = json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}
    ws = session.workspace
    meta = {
        "session_id": session.id,
        "created_at": existing.get("created_at") or _now(),
        "updated_at": _now(),
        "tables": ws.table_names,
        "models": {t: list(ws.table(t).models) for t in ws.table_names},
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)
