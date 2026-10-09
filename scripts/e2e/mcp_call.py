"""Drive the MCP server over stdio, exactly as an agent's client would.

Launches ``meelu-analytics-mcp --stdio`` from this checkout with a throwaway
``TABULAR_BASE``, then either lists the tools or runs a scripted sequence of
tool calls. Used by the autonomous ship-feature / fix-bug skills to verify a
change through the real protocol rather than by calling Python functions.

    uv run python scripts/e2e/mcp_call.py --list
    uv run python scripts/e2e/mcp_call.py --steps steps.json

``steps.json`` is a list of ``{"tool": name, "args": {...}}``. A string value
``"$session_key"`` in args is replaced by the key returned from the first
``create_session`` call. Exits non-zero if any call errors.
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]


def _substitute(value, key):
    if value == "$session_key":
        return key
    if isinstance(value, dict):
        return {k: _substitute(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, key) for v in value]
    return value


def _payload(result):
    text = "".join(getattr(c, "text", "") for c in result.content)
    try:
        return json.loads(text)
    except ValueError:
        return text


async def _run(args) -> int:
    base = args.base or tempfile.mkdtemp(prefix="meelu-e2e-")
    params = StdioServerParameters(
        command="uv",
        args=["run", "--project", str(ROOT), "meelu-analytics-mcp", "--stdio"],
        env={**os.environ, "TABULAR_BASE": base},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if args.list:
                tools = await session.list_tools()
                for t in tools.tools:
                    print(f"{t.name}: {(t.description or '').splitlines()[0] if t.description else ''}")
                return 0

            steps = json.loads(Path(args.steps).read_text())
            key, failed = None, False
            for step in steps:
                call_args = _substitute(step.get("args", {}), key)
                result = await session.call_tool(step["tool"], call_args)
                payload = _payload(result)
                if step["tool"] == "create_session" and isinstance(payload, dict):
                    key = payload.get("session_key", key)
                failed |= bool(result.isError)
                print(json.dumps({"tool": step["tool"], "is_error": result.isError,
                                  "result": payload}, indent=2, default=str))
            return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list registered tools")
    parser.add_argument("--steps", help="JSON file of tool calls to run in order")
    parser.add_argument("--base", help="TABULAR_BASE for the server (default: new temp dir)")
    args = parser.parse_args()
    if not args.list and not args.steps:
        parser.error("pass --list or --steps")
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
