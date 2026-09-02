# Getting started

## Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/)

There is no separate install step. `uv run` resolves and installs dependencies
into `.venv` on the first invocation.

## 1. Run the server

```bash
git clone https://github.com/shubham303/meelu-analytics-mcp.git
cd meelu-analytics-mcp

export TABULAR_BASE="$HOME/.meelu-open/analytics"
mkdir -p "$TABULAR_BASE"

uv run --project . meelu-analytics-mcp
```

The server prints its address and listens on `http://127.0.0.1:8321/mcp` over
the MCP streamable-HTTP transport.

`TABULAR_BASE` matters more than it looks. It is both where sessions are
persisted *and* the only directory the engine is allowed to read files from —
see [Configuration](configuration.md#the-data-directory-boundary). Point it at a
stable location and put your CSVs there.

To run over stdio instead — for clients that launch the server as a subprocess:

```bash
uv run --project . meelu-analytics-mcp --stdio
```

## 2. Connect an agent

**Claude Code:**

```bash
claude mcp add --transport http meelu-analytics http://127.0.0.1:8321/mcp
```

**Any other MCP client:** point it at `http://127.0.0.1:8321/mcp` using the
streamable HTTP transport.

The server binds to localhost and has **no authentication**. Setting
`MEELU_ANALYTICS_HOST=0.0.0.0` exposes every tool — including `run_sql` — to
anyone who can reach the port. Only do that when you control the network.

## 3. A first analysis

Put a CSV inside `TABULAR_BASE`, then, from your agent:

```
create_session(paths=["/Users/you/.meelu-open/analytics/orders.csv"])
→ {"session_key": "s_a1b2c3", "tables": ["orders"], "relationships": {...}}
```

Every later call carries that `session_key`. The data is never re-sent.

```
profile(session_key="s_a1b2c3", table="orders")
```

`profile` is the natural second call: it reports per-column type, missingness,
cardinality, and distribution, which tells you what is worth asking next.

Columns that arrive as unrefined "categorical" should be typed before any
statistical test, because typing drives method selection:

```
list_categorical_columns(session_key="s_a1b2c3", table="orders")
set_column_type(session_key="s_a1b2c3", table="orders",
                column="tier", type="categorical_ordinal")
```

See [Column typing](tools/column-typing.md). Then ask a real question:

```
analyze_association(session_key="s_a1b2c3", table="orders",
                    col_a="tier", col_b="revenue")
```

The engine checks normality and equal variance, routes to a one-way ANOVA or a
Kruskal-Wallis accordingly, and returns the chosen method, the statistic, an
effect size, and a `trust` block. It does not ask you which test you wanted.

## 4. Read the result

Every tool returns the same shape:

```json
{
  "method": "kruskal_wallis",
  "summary": "Revenue differs significantly across tier (H=41.2, p<0.001).",
  "values":   { "statistic": 41.2, "p_value": 0.0000001, "epsilon_squared": 0.18 },
  "metadata": { "normality": "failed", "groups": 4, "n": 812 },
  "trust":    { "level": "high", "caveats": [], "basis": ["n=812"], "declined": false }
}
```

`declined: true` means the tool refused: the data could not support the
question. Report the refusal — never substitute a number. See the
[Honesty model](honesty-model.md).

## Where to go next

- [Session model](session-model.md) — multiple tables, joins, the one-table rule
- [Tool reference](tools/README.md) — all 45 tools
- [Configuration](configuration.md) — storage layout and optional extras
