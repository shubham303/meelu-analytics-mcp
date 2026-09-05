#!/usr/bin/env python3
"""meelu-analytics-mcp installer — detects your agents and wires the server into them.

Normally reached through `install.sh`, which guarantees `uv` is present first.
Standard library only, so it runs under any Python 3.9+.

    python3 scripts/install.py                      # interactive
    python3 scripts/install.py --agent claude-code  # non-interactive
    python3 scripts/install.py --all --yes

The server itself is never cloned or pip-installed: every agent is configured to
launch `uvx meelu-analytics-mcp --stdio`, which resolves the package and all of
its dependencies on first run and caches them afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SERVER_NAME = "meelu-analytics"
PYPI_NAME = "meelu-analytics-mcp"
BIN_NAME = "meelu-analytics-mcp"  # console_script from pyproject [project.scripts]
GIT_URL = "git+https://github.com/shubham303/meelu-analytics-mcp"
DEFAULT_DATA_DIR = Path.home() / "meelu-data"

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
APPDATA = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))


# ---------------------------------------------------------------- pretty ----
class C:
    on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    B = "\033[1m" if on else ""
    D = "\033[2m" if on else ""
    G = "\033[32m" if on else ""
    Y = "\033[33m" if on else ""
    R = "\033[31m" if on else ""
    X = "\033[0m" if on else ""


def say(msg: str = "") -> None:
    print(f"  {msg}" if msg else "")


def ok(msg: str) -> None:
    say(f"{C.G}✓{C.X} {msg}")


def warn(msg: str) -> None:
    say(f"{C.Y}!{C.X} {msg}")


def fail(msg: str) -> None:
    say(f"{C.R}✗{C.X} {msg}")


def ask(prompt: str, default: str = "") -> str:
    """Read from the terminal even when stdin is a pipe (curl | sh)."""
    try:
        if sys.stdin.isatty():
            return input(f"  {prompt}").strip() or default
        with open("/dev/tty") as tty:
            sys.stdout.write(f"  {prompt}")
            sys.stdout.flush()
            return (tty.readline().strip() or default)
    except (OSError, EOFError, KeyboardInterrupt):
        return default


# ------------------------------------------------------------ config i/o ----
def read_json(path: Path) -> dict:
    """Existing config, or {} — a malformed file is backed up rather than lost."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        backup(path)
        warn(f"{path} was not valid JSON — backed it up and started fresh")
        return {}


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".meelu-backup")
    shutil.copy2(path, bak)
    return bak


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".meelu-tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def nested(data: dict, key: str) -> dict:
    """Get-or-create a dict under `key`, tolerating a null/garbage existing value."""
    cur = data.get(key)
    if not isinstance(cur, dict):
        cur = {}
        data[key] = cur
    return cur


# ------------------------------------------------------------- the entry ----
def pypi_available() -> bool:
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{PYPI_NAME}/json", timeout=6
        ) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def tool_path(binary: str) -> str | None:
    """Find `binary` where it lives even when it is not on this shell's PATH."""
    found = shutil.which(binary)
    if found:
        return found
    for d in (Path.home() / ".local/bin", Path.home() / ".cargo/bin"):
        cand = d / (binary + (".exe" if IS_WIN else ""))
        if cand.exists():
            return str(cand)
    return None


def warm_up(binary: str) -> None:
    """Take the first-launch cost here rather than in the user's agent.

    The very first start compiles numba/SHAP kernels — around 90 seconds. An
    agent waiting on the MCP handshake would give up long before that, and the
    server would look broken. Running it once now (stdio with no stdin, so it
    exits at EOF) leaves every later start at a second or two.
    """
    say(f"{C.D}Warming it up, so your assistant doesn't wait on the first run…{C.X}")
    try:
        subprocess.run([binary, "--stdio"], stdin=subprocess.DEVNULL,
                       capture_output=True, timeout=900, check=False)
        ok("ready")
    except subprocess.TimeoutExpired:
        warn("warm-up is taking unusually long — carrying on anyway")
    except OSError as exc:
        warn(f"could not warm it up ({exc}); the first analysis will be slower")


def install_server(source: str) -> tuple[str, list[str]]:
    """Install the server once, and return the command an agent should launch.

    `uv tool install` puts a real `meelu-analytics-mcp` executable on disk with
    its own pinned environment. Agents then start that binary directly — no
    resolution, no network, no wait. Running it through `uvx` instead would make
    every single agent startup re-check the package, which for a git source
    means a fetch each time and MCP connections that time out.
    """
    uv = tool_path("uv") or "uv"
    spec = [PYPI_NAME] if source == "pypi" else ["--from", GIT_URL, PYPI_NAME]
    say(f"{C.D}Installing the server and its dependencies (a minute, once)…{C.X}")
    res = subprocess.run(
        [uv, "tool", "install", "--force", *spec],
        capture_output=True, text=True, check=False,
    )
    if res.returncode == 0:
        binary = tool_path(BIN_NAME)
        if binary:
            ok(f"installed: {binary}")
            warm_up(binary)
            return binary, ["--stdio"]
        warn("installed, but the executable is not where expected — falling back to uvx")
    else:
        warn("`uv tool install` failed — agents will fetch on demand instead")
        say(f"{C.D}{(res.stderr or '').strip()[-300:]}{C.X}")

    # Fallback: let uvx resolve at launch. Slower to start, but it works.
    uvx = tool_path("uvx") or "uvx"
    return uvx, ([PYPI_NAME, "--stdio"] if source == "pypi"
                 else ["--from", GIT_URL, PYPI_NAME, "--stdio"])


# ------------------------------------------------------------- installers ---
# Each installer takes (cmd, args, env) and returns a note for the user, or
# raises. `detected` decides whether the agent shows up pre-ticked in the menu.

def json_installer(path: Path, container: str, style: str = "mcpServers"):
    def install(cmd, args, env):
        data = read_json(path)
        servers = nested(data, container)
        if style == "vscode":
            servers[SERVER_NAME] = {
                "type": "stdio", "command": cmd, "args": args, "env": env,
            }
        elif style == "opencode":
            servers[SERVER_NAME] = {
                "type": "local", "command": [cmd, *args],
                "environment": env, "enabled": True,
            }
        else:
            servers[SERVER_NAME] = {"command": cmd, "args": args, "env": env}
        backup(path)
        write_json(path, data)
        return str(path)
    return install


def install_codex(cmd, args, env):
    """Codex keeps MCP servers in ~/.codex/config.toml under [mcp_servers.*]."""
    path = Path.home() / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    header = f"[mcp_servers.{SERVER_NAME}]"
    # Drop any block we wrote before, up to the next top-level table.
    pattern = re.compile(
        r"(?ms)^\[mcp_servers\." + re.escape(SERVER_NAME) + r"\][^\[]*"
    )
    text = pattern.sub("", text).rstrip()

    def toml_str(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    block = [
        header,
        f"command = {toml_str(cmd)}",
        "args = [" + ", ".join(toml_str(a) for a in args) + "]",
    ]
    if env:
        block.append(
            "env = { "
            + ", ".join(f"{k} = {toml_str(v)}" for k, v in env.items())
            + " }"
        )
    backup(path)
    path.write_text(
        (text + "\n\n" if text else "") + "\n".join(block) + "\n", encoding="utf-8"
    )
    return str(path)


def install_claude_code(cmd, args, env):
    """Prefer the CLI so Claude Code owns the file format; fall back to ~/.claude.json."""
    claude = shutil.which("claude")
    if claude:
        subprocess.run(
            [claude, "mcp", "remove", "-s", "user", SERVER_NAME],
            capture_output=True, check=False,
        )
        argv = [claude, "mcp", "add", "-s", "user", SERVER_NAME]
        for k, v in env.items():
            argv += ["--env", f"{k}={v}"]
        argv += ["--", cmd, *args]
        res = subprocess.run(argv, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return "claude mcp add (user scope)"
        warn(f"`claude mcp add` failed: {res.stderr.strip()[:200]} — writing the config file instead")
    return json_installer(Path.home() / ".claude.json", "mcpServers")(cmd, args, env)


def claude_desktop_path() -> Path:
    if IS_MAC:
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if IS_WIN:
        return APPDATA / "Claude/claude_desktop_config.json"
    return Path.home() / ".config/Claude/claude_desktop_config.json"


def vscode_path(app: str = "Code") -> Path:
    if IS_MAC:
        return Path.home() / f"Library/Application Support/{app}/User/mcp.json"
    if IS_WIN:
        return APPDATA / f"{app}/User/mcp.json"
    return Path.home() / f".config/{app}/User/mcp.json"


class Agent:
    def __init__(self, key, label, install, detect, note=""):
        self.key, self.label = key, label
        self.install, self.detect, self.note = install, detect, note

    @property
    def detected(self) -> bool:
        try:
            return bool(self.detect())
        except OSError:
            return False


def exists(*paths: Path):
    return lambda: any(p.exists() for p in paths)


def on_path(binary: str):
    return lambda: shutil.which(binary) is not None


AGENTS = [
    Agent("claude-code", "Claude Code", install_claude_code,
          lambda: shutil.which("claude") is not None or (Path.home() / ".claude.json").exists(),
          "restart `claude`, then run /mcp to confirm"),
    Agent("claude-desktop", "Claude Desktop",
          json_installer(claude_desktop_path(), "mcpServers"),
          exists(claude_desktop_path(), claude_desktop_path().parent),
          "quit and reopen Claude Desktop"),
    Agent("codex", "OpenAI Codex", install_codex,
          lambda: shutil.which("codex") is not None or (Path.home() / ".codex").exists(),
          "Codex asks you to trust the server the first time it starts"),
    Agent("cursor", "Cursor",
          json_installer(Path.home() / ".cursor/mcp.json", "mcpServers"),
          exists(Path.home() / ".cursor"),
          "reload Cursor, then check Settings → MCP"),
    Agent("windsurf", "Windsurf",
          json_installer(Path.home() / ".codeium/windsurf/mcp_config.json", "mcpServers"),
          exists(Path.home() / ".codeium/windsurf"),
          "hit refresh in the Windsurf MCP panel"),
    Agent("vscode", "VS Code (Copilot)",
          json_installer(vscode_path(), "servers", style="vscode"),
          exists(vscode_path(), vscode_path().parent, vscode_path().parent.parent),
          "reload the window; the server appears in Copilot's tool picker"),
    Agent("gemini-cli", "Gemini CLI",
          json_installer(Path.home() / ".gemini/settings.json", "mcpServers"),
          exists(Path.home() / ".gemini"), "run /mcp inside gemini"),
    Agent("grok", "Grok CLI",
          json_installer(Path.home() / ".grok/user-settings.json", "mcpServers"),
          lambda: shutil.which("grok") is not None or (Path.home() / ".grok").exists(),
          "restart the grok CLI"),
    Agent("opencode", "OpenCode",
          json_installer(Path.home() / ".config/opencode/opencode.json", "mcp", style="opencode"),
          exists(Path.home() / ".config/opencode"), "restart opencode"),
    Agent("zed", "Zed",
          json_installer(Path.home() / ".config/zed/settings.json", "context_servers"),
          exists(Path.home() / ".config/zed"), "restart Zed"),
]
BY_KEY = {a.key: a for a in AGENTS}


# ------------------------------------------------------------------ main ----
def choose(agents: list[Agent]) -> list[Agent]:
    say(f"{C.B}Which assistants should I set this up for?{C.X}")
    say()
    for i, a in enumerate(agents, 1):
        mark = f"{C.G}found on this machine{C.X}" if a.detected else f"{C.D}not detected{C.X}"
        say(f"  {i:>2}. {a.label:<22} {mark}")
    say()
    detected = [a for a in agents if a.detected]
    default = ",".join(str(agents.index(a) + 1) for a in detected) or "1"
    say(f"{C.D}Numbers separated by commas, or 'all'.{C.X}")
    reply = ask(f"Choice [{default}]: ", default).lower()

    if reply in ("all", "a", "*"):
        return agents
    picked = []
    for part in re.split(r"[,\s]+", reply):
        if part.isdigit() and 1 <= int(part) <= len(agents):
            a = agents[int(part) - 1]
            if a not in picked:
                picked.append(a)
        elif part in BY_KEY and BY_KEY[part] not in picked:
            picked.append(BY_KEY[part])
    return picked


def main() -> int:
    p = argparse.ArgumentParser(description="Install meelu-analytics-mcp into your AI agents.")
    p.add_argument("--agent", action="append", default=[],
                   help=f"agent key, repeatable or comma-separated ({', '.join(BY_KEY)})")
    p.add_argument("--all", action="store_true", help="every supported agent")
    p.add_argument("--data-dir", help="folder the server may read (default ~/meelu-data)")
    p.add_argument("--source", choices=["auto", "pypi", "git"], default="auto",
                   help="where to fetch the package from")
    p.add_argument("--yes", "-y", action="store_true", help="no prompts; accept defaults")
    p.add_argument("--print-config", action="store_true",
                   help="print the MCP JSON entry and exit, configuring nothing")
    args = p.parse_args()

    source = args.source
    if source == "auto":
        source = "pypi" if pypi_available() else "git"
    if args.print_config:
        binary = tool_path(BIN_NAME) or BIN_NAME
        print(json.dumps({"mcpServers": {SERVER_NAME: {
            "command": binary, "args": ["--stdio"],
            "env": {"TABULAR_BASE": str(args.data_dir or DEFAULT_DATA_DIR)},
        }}}, indent=2))
        return 0

    say()
    say(f"{C.B}meelu-analytics-mcp{C.X} — deterministic data analysis for your assistant")
    say()
    if source == "git":
        say(f"{C.D}Installing from GitHub (not yet on PyPI).{C.X}")

    # --- data folder ---
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else None
    if data_dir is None:
        say(f"{C.B}Where do your data files live?{C.X}")
        say(f"{C.D}This is the only folder the server is allowed to read.{C.X}")
        reply = str(DEFAULT_DATA_DIR) if args.yes else ask(f"Folder [{DEFAULT_DATA_DIR}]: ", str(DEFAULT_DATA_DIR))
        data_dir = Path(reply).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    ok(f"data folder: {data_dir}")
    say()

    # --- pick agents ---
    keys = [k for spec in args.agent for k in re.split(r"[,\s]+", spec) if k]
    if args.all:
        targets = list(AGENTS)
    elif keys:
        unknown = [k for k in keys if k not in BY_KEY]
        if unknown:
            fail(f"unknown agent(s): {', '.join(unknown)}")
            say(f"known: {', '.join(BY_KEY)}")
            return 2
        targets = [BY_KEY[k] for k in keys]
    elif args.yes:
        targets = [a for a in AGENTS if a.detected]
    else:
        targets = choose(AGENTS)

    if not targets:
        warn("nothing selected — no changes made")
        return 1

    # --- install the server itself, then point every agent at it ---
    say()
    cmd, argv = install_server(source)
    say()
    env = {"TABULAR_BASE": str(data_dir)}
    done, failed = [], []
    for a in targets:
        try:
            where = a.install(cmd, argv, env)
            ok(f"{a.label} {C.D}→ {where}{C.X}")
            done.append(a)
        except Exception as exc:  # a bad path or permissions shouldn't sink the rest
            fail(f"{a.label}: {exc}")
            failed.append(a)

    say()
    say(f"{C.B}Done.{C.X}")
    for a in done:
        if a.note:
            say(f"  • {a.label}: {a.note}")
    say()
    say(f"Put a CSV in {C.B}{data_dir}{C.X}, then ask your assistant:")
    say(f'  {C.D}"Using meelu, load orders.csv and tell me what\'s in it."{C.X}')
    say()
    if failed:
        say(f"{C.D}For anything that failed, `--print-config` gives you the JSON to paste in.{C.X}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  cancelled")
        sys.exit(130)
