# The honesty model

Every result carries a `trust` block, and any tool may refuse to answer.

## Why it is built in

This engine is a trust product. Someone uses it precisely because they cannot run
the analysis themselves — which means they also cannot check the number that
comes back. A confidently-worded wrong answer is therefore worse than no answer,
and the cost of that asymmetry rises with the stakes: a bad causal estimate can
justify a bad decision for months.

So confidence is not an afterthought bolted onto the interesting tools. It is
part of the return contract of all 45.

## The trust block

```json
"trust": {
  "level": "moderate",
  "caveats": ["Moderate sample (68 observations) — reasonable but not definitive."],
  "basis": ["n=68"],
  "declined": false,
  "decline_reason": null
}
```

| Field | Meaning |
|---|---|
| `level` | `high` / `moderate` / `low` / `none` / `unassessed` |
| `caveats` | Plain-language warnings meant to be shown to the user verbatim |
| `basis` | What drove the level — e.g. `n=68`, `silhouette=0.31` |
| `declined` | The tool refused to answer |
| `decline_reason` | Why it refused |

### The levels

| Level | Meaning |
|---|---|
| `high` | Ample data, assumptions hold, quality signals are good |
| `moderate` | Usable but not definitive — report it with its caveats |
| `low` | Directional only. Do not build a decision on this number alone |
| `none` | Paired with `declined: true` — there is no number |
| `unassessed` | Confidence has not been judged for this method yet |

`unassessed` is the honest default, and that is the whole point of having it. A
result whose confidence nobody has evaluated says so, rather than defaulting to
looking confident. It is an admission, not an endorsement.

## How levels are assigned

Sample size is the floor: under 30 usable observations is `low`, under 100 is
`moderate`, above that `high`. Individual methods then add their own signals and
caveats on top — silhouette score for clustering, explained variance for PCA,
assumption-check outcomes for statistical tests, refutation outcome for causal
estimates.

When a result depends on several assessments, they combine with **the most
cautious level winning**, caveats unioned, and a single decline dominating
everything. Confidence never launders itself upward by averaging.

## Declines

When the data cannot support the question, the tool returns a decline instead of
a number. Real examples from the engine:

- Fewer than 10 usable rows for an association test.
- Fewer than 30 rows, or a class with fewer than 2 examples, for model training —
  a model whose metrics are meaningless is worse than a refusal.
- Fewer than ~12 points, or under two seasonal cycles, for a forecast.
- A single-valued treatment for a causal estimate — there is nothing to compare.
- A failed placebo refutation for a causal estimate. The effect estimate is
  withheld *entirely*, not returned with a warning.

A refusal is a stronger signal than a meaningless number. It is information about
the data, and it is usually the most valuable thing the tool can say.

## What an agent must do with this

The server's own instructions tell connected models:

> When `declined` is true the data cannot support the question: report the refusal
> and its reason and do NOT substitute a number. Always convey the trust level
> and caveats to the user; never present a low-trust or declined result as a
> confident fact.

Concretely:

- **Never** substitute an estimate, a rule of thumb, or a different tool's output
  for a declined result without saying that is what you are doing.
- **Always** surface caveats. They are written for the end user, not for you.
- **Never** report a `low` or `unassessed` result in the same voice as a `high`
  one.

## Observational data never earns high trust

Causal estimates from observational data are capped at `moderate`, no matter how
clean the data or how well the refutation passes. Backdoor adjustment can only
control for confounders you named and measured; unmeasured confounding is
invisible to the method and to the refutation. The ceiling encodes that
limitation so it does not have to be remembered.
