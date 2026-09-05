<!-- mcp-name: io.github.shubham303/meelu-analytics-mcp -->

# meelu-analytics-mcp

**Ask your AI assistant real questions about your data — and get answers you can
actually rely on.**

You have a spreadsheet. You want to know what's in it, what's driving a number,
who your best customers are, where things are heading. So you ask Claude.

Today it will write a little program on the spot and give you an answer. Usually
that's fine. When it isn't, you can't tell: it might have used the wrong kind of
test for your data, or graded its own prediction on the same rows it learned
from, or told you one thing causes another when it only happens alongside it. The
answer looks equally confident either way.

This tool sits between your assistant and your data and takes those judgement
calls out of its hands. You ask the question in plain English; a proper analysis
engine works out the right method from your actual data, runs it, and tells you
how much to trust the result — or says the data can't answer the question, rather
than making something up.

## What it looks like

<p align="center">
  <img src="docs/assets/example-chat.svg" alt="A chat exchange: the user asks which customer tier drives revenue and whether the difference is real. The assistant loads the files, joins them, and runs an association test; the engine finds the data isn't normally distributed and switches to a Kruskal-Wallis test, then returns a revenue-by-tier table with a trust rating of 'high' based on 4,812 rows." width="900">
</p>

You asked a question. You didn't have to know that comparing four groups against
a revenue figure calls for a particular statistical test, or that this data broke
the assumptions of the obvious one and needed a different test instead. That was
worked out from the data, and written down in the answer — so if someone asks six
months from now how you got that number, it's on the record.

## What you can ask

There are 45 tools under the hood, but you never call them. You ask; your
assistant picks. In practice that means questions like:

| You ask | What you get back |
|---|---|
| *"What's in this file?"* | Every column explained — what kind of data it holds, what's missing, what's typical, and which "numbers" are really just ID codes |
| *"Are these two things related?"* | A real statistical test, chosen to suit your data — and how big the relationship is, not just whether it's detectable |
| *"What's driving revenue?"* | The factors that matter most, ranked, plus plain rules you can act on: *"highest among premium customers who've been with us over 18 months"* |
| *"Do my customers fall into groups?"* | Natural segments found in the data, each described so you can tell what makes it distinct |
| *"What predicts churn?"* | A prediction model, scored honestly on data it was never shown, plus what it's really paying attention to |
| *"Where is this heading?"* | The underlying trend separated from seasonal ups and downs, a forecast with honest margins of error, and the dates when behaviour shifted |
| *"Did the price change cause this?"* | A genuine attempt at cause and effect — put through a sanity check, and withheld entirely if it fails |
| *"Who are my best customers?"* | Customer segments by how recently and often they buy, how well you retain them over time, and what tends to get bought together |

Related files can be analysed together, and anything the built-in tools don't
cover, you can ask for directly — it becomes part of the same analysis.

## Getting set up

One command. It asks which assistant you use, sets it up, and downloads
everything it needs.

Open **Terminal** (on Mac: `⌘ Space`, type "Terminal", press enter) and paste:

```bash
curl -fsSL https://raw.githubusercontent.com/shubham303/meelu-analytics-mcp/main/install.sh | sh
```

You'll be asked two things — where meelu should keep its sessions (press enter
for the default, a `meelu-data` folder in your home directory) and which
assistants to connect. It detects the ones you already have installed and pre-selects them:

```
  Which assistants should I set this up for?

   1. Claude Code            found on this machine
   2. Claude Desktop         found on this machine
   3. OpenAI Codex           not detected
   4. Cursor                 found on this machine
   ...

  Choice [1,2,4]:
```

Then restart the assistant you picked, and you're done. There's nothing to leave
running and nothing to remember — your assistant starts the tool when it needs it.

<details>
<summary><b>Supported assistants, and what the installer touches</b></summary>

| Assistant | Config it writes |
|---|---|
| Claude Code | `claude mcp add` (user scope), or `~/.claude.json` |
| Claude Desktop | `claude_desktop_config.json` |
| OpenAI Codex | `~/.codex/config.toml` |
| Cursor | `~/.cursor/mcp.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| VS Code (Copilot) | `<VS Code>/User/mcp.json` |
| Gemini CLI | `~/.gemini/settings.json` |
| Grok CLI | `~/.grok/user-settings.json` |
| OpenCode | `~/.config/opencode/opencode.json` |
| Zed | `~/.config/zed/settings.json` |

Every existing config is copied to a `.meelu-backup` file before it's changed,
and only the `meelu-analytics` entry is added — your other servers are left
alone.

The one prerequisite is [`uv`](https://docs.astral.sh/uv/); if you don't have
it, the installer offers to install it. `uv` then handles Python and all ~15
libraries by itself — there is nothing else to install, and nothing to clone.

The server is installed once, up front, into its own isolated environment
(`uv tool install`), and your agents are pointed at the resulting program. So
starting it is instant and works offline; it isn't downloading anything each
time your assistant wakes up. The installer also runs it once itself — the very
first start compiles some numerical kernels and takes about a minute, and it's
better that happens here than while your assistant is waiting on it.

Non-interactive, for scripting or a second machine:

```bash
curl -fsSL .../install.sh | sh -s -- --agent claude-code,codex --data-dir ~/data --yes
```

</details>

<details>
<summary><b>Claude on the web, Cowork, or an assistant that isn't listed</b></summary>

Anything that reads a standard MCP config takes this entry — print it with

```bash
curl -fsSL .../install.sh | sh -s -- --print-config
```

```json
{
  "mcpServers": {
    "meelu-analytics": {
      "command": "/Users/you/.local/bin/meelu-analytics-mcp",
      "args": ["--stdio"],
      "env": { "TABULAR_BASE": "/Users/you/meelu-data" }
    }
  }
}
```

Claude on the web and Claude Cowork run in Anthropic's cloud, so they can't
start a program on your laptop. There the tool has to be reachable over the
network: run `uvx meelu-analytics-mcp` (no `--stdio`) to get an HTTP server on
`http://127.0.0.1:8321/mcp` (run `meelu-analytics-mcp` with no `--stdio`),
expose it, and add that URL as a custom connector.
Your data leaves your machine if you do that — for local files, use one of the
desktop assistants above.

</details>

### Just ask

That's it. Talk to your assistant normally, naming your file wherever it lives:

> Using meelu, load ~/Downloads/orders.csv and tell me what's in it.

> Using meelu, is there a real relationship between the discount we gave and
> whether the order came back? And which test did it use?

> Using meelu, load orders.csv and customers.csv, put them together, and tell me
> what predicts whether a customer leaves.

Saying **"using meelu"** the first time steers your assistant to this tool
instead of improvising its own answer. After that it'll carry on using it.

Name as many files as you like. If several of them are related — orders and
customers, say — the tool will spot how they connect on its own. *Working with
Excel? Save as CSV first: File → Save As → CSV.*

Three follow-ups worth having in your back pocket, because they're where this
tool earns its keep:

- **"Which test did it use, and why?"** — there's always an answer, and it's
  recorded.
- **"How much should I trust this?"** — every result carries a rating. If the
  tool refused to answer, *that's the answer* — don't let your assistant fill the
  gap with a guess.
- **"Is that actually causing it, or just related?"** — almost always the latter,
  and the difference matters enormously before you act on it.

→ **[Getting started](docs/getting-started.md)** walks through the same ground in
more depth.

## Why it's built this way

Three decisions do most of the work.

**The method is chosen from your data, not guessed at.** Comparing groups against
a number is usually one particular test — unless your data doesn't meet that
test's assumptions, in which case it needs a different one, and the difference
changes the answer. That check runs every single time, and what it found is
reported alongside the result.

**It's willing to say no.** Too few rows to be meaningful. A prediction that
would be graded on the rows it learned from. A cause-and-effect claim that fails
its own sanity check. In each case you get a plain explanation instead of a
number. A refusal tells you something true; a confident wrong answer doesn't.

**Your work builds up.** Segments, predictions, and any new columns you create
get saved back alongside your data, so each question builds on the last instead
of starting from scratch. It all survives shutting down and coming back
tomorrow.

## Documentation

The technical details, for when you want them:

| Guide | What's in it |
|---|---|
| [Getting started](docs/getting-started.md) | Install, run the server, connect an agent, first analysis |
| [Configuration](docs/configuration.md) | Environment variables, storage layout, optional extras |
| [Session model](docs/session-model.md) | Sessions, tables, the one-table rule, persistence |
| [Honesty model](docs/honesty-model.md) | Trust levels, caveats, declines — and how to read them |
| [Association test selection](docs/association-tests.md) | The deterministic routing table, in detail |
| [Architecture](docs/architecture.md) | Module layout and the dependency rules behind it |
| **[Tool reference](docs/tools/README.md)** | All 45 tools, by category |

### Tool reference by category

- [Session & workspace](docs/tools/session-and-workspace.md) — ingest, join, SQL, table building
- [Column typing](docs/tools/column-typing.md) — the typing that drives statistical routing
- [Descriptive & exploratory](docs/tools/descriptive.md) — profile, outliers, association
- [Feature engineering](docs/tools/feature-engineering.md) — deterministic column builders
- [Clustering & dimensionality reduction](docs/tools/clustering-and-dimreduction.md)
- [Supervised machine learning](docs/tools/supervised-ml.md) — train, evaluate, explain
- [Time series](docs/tools/time-series.md) — decompose, forecast, changepoints
- [Drivers & causal inference](docs/tools/drivers-and-causal.md)
- [Customer analytics](docs/tools/customer-analytics.md) — basket, RFM, cohorts

Built on DuckDB, scikit-learn, statsmodels, SHAP and DoWhy. Runs entirely on your
own machine — your data never leaves it.

## Roadmap

Everything below ships today.

| | Capability |
|---|---|
| ✅ | **Explore** — summarise any file, flag unusual values, scan for relationships |
| ✅ | **Test** — statistical tests chosen automatically from the data's shape |
| ✅ | **Segment** — find natural groupings and describe what makes each distinct |
| ✅ | **Predict** — train models, score them honestly, explain what drives them |
| ✅ | **Forecast** — separate trend from seasonality, project forward, find turning points |
| ✅ | **Explain** — rank what's driving a number; estimate cause and effect |
| ✅ | **Customers** — retention cohorts, RFM segments, market basket analysis |
| ✅ | **Build** — engineer new columns, clean messy tables, query with SQL |

### Coming next

- **Bigger files** — sampling and out-of-memory strategies, so large data degrades
  gracefully rather than being refused.
- **Windows** — the installer is tested on macOS and Linux; Windows support is
  next.
- **More file types** — Excel, Parquet and JSON alongside CSV.
- **Better trust ratings** — real confidence assessments everywhere they're still
  missing.
- **Fewer manual steps** — analysing related files without joining them by hand.

### Deliberately not doing

Dashboards, reports, and connecting to your other systems. This is the analysis
engine; what you do with the answer is up to you.

## Contributing

Issues and pull requests are welcome. When adding an analytic, keep the two
invariants: method selection must be deterministic and recorded in the result's
metadata, and every result must carry an honest `trust` block — including a
decline when the data cannot support the question. See
[Architecture](docs/architecture.md) for where things belong.

```bash
uv sync --extra dev --extra insights
uv run pytest
```

## Credits and license

Ported from [TableIntelligence](https://github.com/shubham303/TableIntelligence)
by the same author. Licensed under the [MIT License](LICENSE).
