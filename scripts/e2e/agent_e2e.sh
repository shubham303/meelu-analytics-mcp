#!/usr/bin/env bash
# Ask a real headless Claude agent a question that it can only answer through
# this checkout's MCP server, and print what it did.
#
#   scripts/e2e/agent_e2e.sh "Using meelu-analytics, profile /abs/path/data.csv"
#
# The server runs over stdio from this checkout with a throwaway TABULAR_BASE,
# and the agent may use only the meelu-analytics tools, so a correct answer
# proves the feature works end to end through the protocol.
set -euo pipefail

question="${1:?usage: agent_e2e.sh \"<question that needs the meelu-analytics tools>\"}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d -t meelu-agent-e2e)"

cat >"$work/mcp.json" <<EOF
{
  "mcpServers": {
    "meelu-analytics": {
      "command": "uv",
      "args": ["run", "--project", "$root", "meelu-analytics-mcp", "--stdio"],
      "env": {"TABULAR_BASE": "$work/base"}
    }
  }
}
EOF
mkdir -p "$work/base"

claude -p "$question" \
  --mcp-config "$work/mcp.json" \
  --strict-mcp-config \
  --allowedTools "mcp__meelu-analytics__*" \
  --output-format stream-json --verbose \
  >"$work/transcript.jsonl"

# Tool calls the agent made, then its final answer.
echo "== tool calls =="
grep -o '"name":"mcp__meelu-analytics__[a-z_]*"' "$work/transcript.jsonl" | sort | uniq -c || true
echo "== final answer =="
tail -n 1 "$work/transcript.jsonl" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("result",""))'
echo "== transcript: $work/transcript.jsonl"
