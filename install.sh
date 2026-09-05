#!/bin/sh
# meelu-analytics-mcp — one-command installer.
#
#   curl -fsSL https://raw.githubusercontent.com/shubham303/meelu-analytics-mcp/main/install.sh | sh
#
# This script is only a bootstrap: it makes sure `uv` exists (that is the whole
# dependency story — uv fetches Python and every library on first run) and then
# hands over to scripts/install.py, which detects your agents, asks which ones
# to wire up, and writes their config files.
set -eu

REPO="${MEELU_REPO:-shubham303/meelu-analytics-mcp}"
BRANCH="${MEELU_BRANCH:-main}"
RAW="https://raw.githubusercontent.com/$REPO/$BRANCH"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

say ""
say "  meelu-analytics-mcp installer"
say "  ─────────────────────────────"
say ""

# --- uv -----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  # uv puts itself here; a previous install may just not be on this shell's PATH.
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -x "$d/uv" ] && PATH="$d:$PATH" && export PATH && break
  done
fi

if ! command -v uv >/dev/null 2>&1; then
  say "  uv is not installed. It handles Python and every dependency for you."
  if [ -t 0 ]; then
    printf '  Install it now from https://astral.sh/uv ? [Y/n] '
    read -r reply </dev/tty || reply=y
  else
    # Piped from curl: stdin is the script, so ask on the terminal if there is one.
    if [ -t 1 ] && [ -e /dev/tty ]; then
      printf '  Install it now from https://astral.sh/uv ? [Y/n] '
      read -r reply </dev/tty || reply=y
    else
      reply=y
    fi
  fi
  case "${reply:-y}" in
    [Nn]*) die "uv is required. See https://docs.astral.sh/uv/getting-started/installation/" ;;
  esac
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "could not install uv"
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -x "$d/uv" ] && PATH="$d:$PATH" && export PATH && break
  done
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH — open a new terminal and re-run this"
  say "  ✓ uv installed"
else
  say "  ✓ uv found ($(uv --version 2>/dev/null || echo present))"
fi

# --- hand over to the real installer ------------------------------------
# Run from a checkout if we are in one, otherwise fetch the script.
here="$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || true)"
if [ -n "${here:-}" ] && [ -f "$here/scripts/install.py" ]; then
  exec uv run --no-project --python 3.11 "$here/scripts/install.py" "$@"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM
curl -fsSL "$RAW/scripts/install.py" -o "$tmp/install.py" || die "could not download the installer"
uv run --no-project --python 3.11 "$tmp/install.py" "$@"
