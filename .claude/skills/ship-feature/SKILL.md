---
name: ship-feature
description: Autonomously pick the next pending item from docs/BACKLOG.md, research the best approach, plan, implement, test end-to-end through the MCP server with a real agent, and open a PR. Use when asked to "ship the next feature", "work the backlog", or for the daily autonomous feature run. Optional argument - a backlog ID (e.g. B-007) to work on instead of the top item.
---

# Ship one backlog item, end to end, without a human

You are the daily feature agent for **meelu-analytics-mcp**. Finish one backlog
item completely: research → plan → review the plan → implement → test (unit +
MCP protocol + real agent) → review the code → PR. Never ask the user
questions; make the call yourself and write down why in the PR. Never merge,
never push to `main`, never force-push.

## 0. Preflight

```bash
git fetch origin && git checkout main && git pull --ff-only
gh label create autonomous --color 5319e7 --description "Opened by an autonomous agent" 2>/dev/null || true
uv sync --extra dev --extra insights
uv run pytest -q            # note any failures that already exist on main
```

If the working tree is dirty, stop and report — don't touch someone else's work.

## 1. Pick the item

1. Read `docs/BACKLOG.md`. Candidates have `**Status:** pending`.
2. Remove items already being worked: `gh pr list --state open --label autonomous --json title,headRefName`
   — skip any ID that appears in an open PR's title or branch.
3. If `$ARGUMENTS` names an ID, take that one. Otherwise take the first
   candidate by order (the file is sorted by priority). Prefer items with
   `Effort: S` or `M`. If the top item is `L`, split it: ship a coherent first slice
   now and add the rest to the backlog as new items.
4. Branch: `git checkout -b auto/<id-lowercase>-<short-slug>`.

## 2. Research (do this properly — it decides the quality of the result)

Launch subagents **in parallel, in one message**:
- **Web research** (general-purpose): current best libraries/approaches for the
  item as of today — maintenance status, license (must permit commercial use;
  flag anything else), install weight, Python 3.10–3.12 + macOS/Linux wheels,
  determinism. Ask for cited sources and a recommendation.
- **Codebase exploration** (Explore): where the change belongs (read
  `docs/architecture.md`), the closest existing tool to copy patterns from, how
  results/trust/declined are built (`src/tabint/shared/results.py`,
  `src/tabint/shared/honesty.py`), and the docs page under `docs/tools/` to update.

## 3. Plan, then get it reviewed

Write the plan into your notes (it goes in the PR body later):
files to change, public tool signatures, how method selection stays
**deterministic and recorded in metadata**, how the **trust block** and
**declined** path behave, new deps (optional extras + lazy import, like
`insights`), tests, and the exact end-to-end checks you'll run.

Then launch a **Plan** subagent to critique it against the codebase and the
two invariants. Address its points before writing code.

## 4. Implement

- Match the surrounding code: naming, comment density, idioms, result shapes.
- Heavy or niche deps go in an optional extra and are imported lazily; a missing
  extra must produce an honest declined result, not a traceback.
- Update `docs/tools/*.md`, the tool table in `README.md`, and
  `scripts/install.py` if install behaviour changes.
- Add pytest tests in `tests/` (happy path, declined/edge path, determinism —
  same input twice gives same method and output).

## 5. Verify end to end (all three must pass)

1. **Unit:** `uv run pytest -q` — no new failures versus preflight.
2. **Protocol:** write a steps file into a temp dir and run
   `uv run python scripts/e2e/mcp_call.py --steps <file>` (see its docstring;
   `"$session_key"` is substituted for you). Use a realistic generated CSV
   (write it with pandas/numpy in a temp dir) and include an edge case that should
   decline. Check: no `is_error`, `metadata` records the method, a `trust`
   block exists, numbers match an independent pandas/scipy computation.
3. **Real agent:** `scripts/e2e/agent_e2e.sh "<plain-English question a user
   would ask that needs the new feature, with the absolute CSV path>"`. Confirm
   the agent called the new/changed tool and the answer is correct and conveys
   trust. If the `claude` CLI isn't available (some cloud runners), say so in
   the PR and treat step 2 as the protocol-level proof instead.

If something fails: debug, fix, re-run all three. If after a serious effort it
still can't be made to work, open the PR as **draft**, explain the blocker
precisely, and set the backlog item to `blocked`.

## 6. Independent code review

Launch a general-purpose subagent with only: the backlog item text, `git diff
main...HEAD`, and the instruction to find correctness bugs, invariant
violations (non-deterministic selection, missing/dishonest trust), missing
tests, and style drift. Fix every real finding; re-run step 5 if code changed.

## 7. Update the backlog and ship

- In `docs/BACKLOG.md` set the item to `**Status:** in-review` and append
  `- **PR:** <filled after creation>`; add any follow-ups you discovered as new
  `pending` items with the next free ID, in priority position.
- Commit (conventional message, e.g. `feat(<area>): ...`) ending with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- `git push -u origin HEAD`
- `gh pr create --base main --label autonomous --title "<ID>: <title>" --body ...`
  Body sections: **Backlog item**, **Research** (recommendation + sources),
  **Design / decisions** (including anything you decided without asking),
  **Verification** (pytest summary, protocol output excerpt, agent question +
  tool calls + answer excerpt), **Limitations / follow-ups**. End the body with:
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
- Put the PR number into the backlog `PR:` line, commit, push.

## 8. Report

Finish with: item ID and title, PR URL, what shipped, how it was verified, and
anything a human should look at before merging.
