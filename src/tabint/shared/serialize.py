"""Shared JSON serialization for the CLI and MCP surfaces.

Both front-ends must turn Result objects (and arbitrary values/metadata dicts that
may contain numpy scalars or NaN) into strict-JSON-safe structures. Keeping that in
one place means the two surfaces never diverge on how a result is rendered.
"""
from __future__ import annotations

import datetime as _dt
import math
from decimal import Decimal
from typing import Any

import numpy as np


def jsonable(obj: Any) -> Any:
    """Recursively coerce numpy scalars, NaN, and ±Infinity into JSON-safe values.

    NaN and infinities are not valid JSON (RFC 8259) and are rejected by strict
    parsers (Node, jq, Go, Rust), so they become null — an agent must never get a
    ``0``-exit success that it then fails to parse.
    """
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    # DuckDB DECIMAL columns come back as Python Decimal, and DATE/TIMESTAMP as
    # datetime objects — none are JSON-serializable, so coerce them here.
    if isinstance(obj, Decimal):
        obj = float(obj)
    elif isinstance(obj, (_dt.date, _dt.datetime, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def result_dict(res: Any) -> dict:
    """Render a Result as a plain JSON dict (dropping any non-serializable artifact).

    Always emits the honesty envelope — a ``trust`` block and a top-level
    ``declined`` flag — so every tool's output carries confidence uniformly. An
    analytic that hasn't set trust reports ``unassessed`` (never a fake ``high``).
    """
    from . import honesty

    trust = getattr(res, "trust", None) or honesty.unassessed()
    trust_json = trust.model_dump(mode="json")
    return {
        "method": res.method,
        "summary": res.summary,
        "trust": trust_json,
        "declined": bool(trust_json.get("declined", False)),
        "values": jsonable(res.values),
        "metadata": jsonable(res.metadata),
    }
