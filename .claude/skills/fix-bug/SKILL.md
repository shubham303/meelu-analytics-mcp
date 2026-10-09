---
name: fix-bug
description: Autonomously pick a random tool of the meelu-analytics MCP server, test it end-to-end through the protocol and with a real agent on realistic and adversarial data, and if anything is broken, find the root cause, fix it, and open a PR. Use when asked to "hunt bugs", "test a random feature", or for the daily autonomous bug run. Optional argument - a tool name to test instead of a random one.
---

# Test one random feature end to end; fix what's broken

You are the daily bug-hunting agent for **meelu-analytics-mcp**. Pick one tool,
try hard to break it the way real users and agents would, and if it's broken,
fix it properly. Never ask the user questions. Never merge, never push to
`main`, never force-push.

## 0. Preflight

```bash
git fetch origin && git checkout main && git pull --ff-only
gh label create autonomous --color 5319e7 --description "Opened by an autonomous agent" 2>/dev/null || true
gh label create bug --color d73a4a 2>/dev/null || true
uv sync --extra dev --extra insights
uv run pytest -q
```

If the working tree is dirty, stop and report. If `pytest` already fails on
`main`, that failure **is** today's bug — go straight to step 3 with it.

## 1. Pick the target

```bash
uv run python scripts/e2e/mcp_call.py --list > /tmp/meelu-tools.txt
gh pr list --state open --label autonomous --json title,headRefName
```

If `$ARGUMENTS` names a tool, use it. Otherwise choose randomly
(`python3 -c 'import random,sys; print(random.choice(open(sys.argv[1]).read().split("\n")[:-1]).split(":")[0])' /tmp/meelu-tools.txt`),
re-rolling if the tool already has an open autonomous PR. Read the tool's
function in `src/tabint/analysis/tools.py`, the algorithm behind it, and its
page under `docs/tools/`, so you know what it promises.

## 2. Try to break it

Generate datasets with pandas/numpy in a temp dir. Always include one realistic
dataset where you **know the right answer** (planted effect, known clusters,
known trend), plus several adversarial ones chosen for this tool, e.g.:
nulls in key columns, a constant column, a single row, two rows, all-unique
IDs, high-cardinality categoricals, unicode/space-laden column names, numeric
strings, mixed date formats, timezone-aware timestamps, duplicate column
names, an empty CSV with only a header, very skewed or tiny classes, huge
values, negative values where only positives make sense.

Run them through the real protocol with
`uv run python scripts/e2e/mcp_call.py --steps <file>` (prerequisite calls like
`create_session`, `join`, `set_column_type` go first; `"$session_key"` is
substituted). Then ask a real agent a natural question that needs the tool:
`scripts/e2e/agent_e2e.sh "<question with absolute CSV path>"`
(if the `claude` CLI is unavailable, note it and rely on the protocol runs).

A run is a **bug** if any of these hold:
- a tool error/traceback where a clear `declined` result was due;
- a wrong number versus your independent pandas/scipy/sklearn computation;
- non-determinism (same input twice → different method or output);
- the result's `metadata` doesn't record the chosen method, or `trust` is
  missing, overconfident (e.g. `high` on 3 rows), or `declined` without a reason;
- the tool description/docs promise something the tool doesn't do, such that
  the real agent misuses it or gives a wrong answer;
- a file-access escape outside `TABULAR_BASE` / the data root.

If you find nothing after a genuine effort (at least 6 distinct scenarios + the
agent run), test **one more** random tool. If both are clean, stop: report what
you tested and the scenarios, and open no PR.

## 3. Fix it

1. `git checkout -b auto/fix-<tool>-<short-slug>`
2. Reproduce first: add a pytest in `tests/` that fails on the current code
   (follow `tests/test_containment.py` for fixtures and the `_tool(name)` helper).
3. Find the **root cause** — don't patch the symptom at the tool boundary if the
   bug lives in the algorithm or a shared helper. Check whether sibling tools
   share the faulty code path and fix them too (with tests).
4. Keep the fix minimal and in the surrounding style. Keep both invariants:
   deterministic method selection recorded in metadata; honest trust/declined.
5. Re-run: `uv run pytest -q`, the failing protocol steps, and the agent
   question. All must now pass.

If you found several bugs, fix the most severe one in this PR and list the
rest in the PR body (and add them to `docs/BACKLOG.md` as `pending`, `Type: bug`).

## 4. Independent review

Launch a general-purpose subagent with the bug description, the reproduction,
and `git diff main...HEAD`; ask it to check the root-cause claim, look for
regressions and missed sibling code paths. Fix real findings and re-verify.

## 5. Ship

- Commit `fix(<area>): <what was wrong>` ending with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- `git push -u origin HEAD`
- `gh pr create --base main --label autonomous --label bug --title "fix(<tool>): ..." --body ...`
  Body: **Tool tested**, **Scenarios tried** (table: scenario → result),
  **Bug** (exact input, expected vs actual), **Root cause**, **Fix**,
  **Verification** (new test, pytest summary, protocol excerpt, agent question
  + answer before/after), **Other issues found**. End the body with:
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`

## 6. Report

Finish with: tool(s) tested, scenarios, bug found (or "none"), PR URL, and
anything a human should double-check.
