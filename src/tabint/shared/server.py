"""The single FastMCP server instance every feature registers its tools onto.

Holds the one ``mcp`` object + the agent instructions string, so that each
feature's ``tools.py`` module can ``@mcp.tool()``-decorate its own functions
without importing from the composition root (``app/``). The dependency arrow
stays one-way: ``app`` imports ``shared``; features import ``shared``; nothing
imports ``app`` except the entry points.

``_get_session`` / ``_SESSIONS`` / ``_BASE`` live here too: they are the
session registry the analysis tools need, and putting them in ``shared`` lets
both ``analysis/tools.py`` and a future transport reach them without a cycle.
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from tabint.analysis.db import persistence
from tabint.analysis.session import Session

_INSTRUCTIONS = """Deterministic single-table data analysis. Workflow:
1. create_session(paths) -> returns a session_key plus the tables and detected
   foreign-key relationships. Pass the session_key to every subsequent tool.
2. Every analytic runs on ONE table (an uploaded table or one produced by join).
   For multiple related tables, call join(session_key, tables) to materialize a
   combined table, then run analytics on it.
3. Each tool returns a structured result: the chosen method, a one-line summary,
   the values (statistics/scores), and metadata (assumptions, params). Trust the
   method it picked — test/algorithm selection is made deterministically.
4. Every result also carries a `trust` block (a confidence level —
   high/moderate/low/none/unassessed — plus caveats) and a `declined` flag. When
   `declined` is true the data cannot support the question: report the refusal and
   its reason and do NOT substitute a number. Always convey the trust level and
   caveats to the user; never present a low-trust or declined result as a
   confident fact."""

mcp = FastMCP("tabint", instructions=_INSTRUCTIONS)

_BASE = os.environ.get("TABULAR_BASE") or "."
_SESSIONS: dict[str, Session] = {}


def get_session(session_key: str) -> Session:
    """Resolve a session by key, reopening from disk on a cache miss."""
    session = _SESSIONS.get(session_key)
    if session is None:
        session = persistence.open_session(session_key, base=_BASE)  # raises if unknown
        _SESSIONS[session_key] = session
    return session


def register_session(session: Session) -> None:
    """Hold a freshly created/loaded session in the live registry."""
    _SESSIONS[session.id] = session


def session_base() -> str:
    return _BASE
