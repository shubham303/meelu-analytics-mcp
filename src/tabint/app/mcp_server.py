"""MCP server entry point — the composition root.

This module is intentionally thin: it does nothing but import the analysis
``tools`` module so its ``@mcp.tool()`` definitions register onto the single
FastMCP instance held in ``tabint.shared.server``, then expose ``main()`` to
run it over streamable HTTP. All tool logic lives in ``analysis/tools.py``.

Run with:  ``uv run meelu-analytics-mcp``  (or ``python -m tabint.app.mcp_server``).
The server listens on ``MEELU_ANALYTICS_HOST``:``MEELU_ANALYTICS_PORT``
(default ``127.0.0.1:8321``) and speaks MCP at ``/mcp``. Set
``TABULAR_BASE`` to control where sessions are stored (default: cwd).
Pass ``--stdio`` to run over stdio instead.
"""
import os
import sys

# Importing this module registers its @mcp.tool() / @mcp.prompt() decorators on
# the shared FastMCP instance.
from tabint.analysis import tools as _analysis_tools  # noqa: F401

from tabint.shared.server import mcp


def main() -> None:
    """Entry point: run the MCP server over streamable HTTP (or stdio with --stdio)."""
    if "--stdio" in sys.argv[1:]:
        mcp.run()
        return

    host = os.environ.get("MEELU_ANALYTICS_HOST", "127.0.0.1")
    port = int(os.environ.get("MEELU_ANALYTICS_PORT", "8321"))
    mcp.settings.host = host
    mcp.settings.port = port
    print(f"meelu-analytics-mcp listening on http://{host}:{port}/mcp", file=sys.stderr)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
