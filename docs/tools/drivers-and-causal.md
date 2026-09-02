# Drivers & causal inference

Two tools that answer superficially similar questions and are epistemically very
far apart. `explain_metric` describes what is *associated* with a metric.
`causal_effect` attempts to estimate what would *change* it. Do not swap them.

## `explain_metric(session_key, table, target, max_depth=3)`

Explain a metric with a shallow decision tree over all other columns: ranked
feature importances plus human-readable segment rules.

The tree is deliberately constrained — depth ≤ 3, and each leaf must hold at
least 5% of the rows. A deeper tree would fit better and explain worse. The
entire point is output a person can read:

> Revenue is highest where `tier = premium` and `tenure_months > 18` (n=412,
> mean £890 vs £310 overall).

That is a segment description, and it is genuinely useful for deciding where to
look next. The 5% leaf floor exists so that no rule is derived from a handful of
rows, which is where interpretable models most often mislead.

**Read the rules as descriptions, not levers.** "Customers who contacted support
churn more" is a true and useful segment. It does not mean support calls cause
churn, and suppressing the support line will not fix retention.

## `causal_effect(session_key, table, treatment, outcome, confounders=None)`

Estimate the causal effect of `treatment` on `outcome` using DoWhy's backdoor
adjustment — linear regression controlling for confounders — followed by a
**random-common-cause placebo refutation**. Needs the
[`insights` extra](../configuration.md#dependencies).

`confounders` defaults to every other usable column. That default is a starting
point, not a substitute for thinking: adjusting for a variable on the causal path
between treatment and outcome (a mediator) removes exactly the effect you are
trying to measure, and adjusting for a collider can manufacture an association
that does not exist. When you know the domain, name the confounders.

### The refutation is not decoration

After estimating, the engine adds a randomly generated common cause and
re-estimates. A valid estimate should barely move. If it does move, the model is
responding to noise, and **the effect estimate is withheld entirely** — not
returned with a warning attached. See the
[Honesty model](../honesty-model.md#declines).

This is the strictest refusal in the engine, on purpose. A causal number is the
kind that gets quoted in a decision meeting six months later, stripped of every
caveat that came with it.

### Observational estimates never earn more than moderate trust

No matter how clean the data or how comfortably the refutation passes,
`causal_effect` is capped at `moderate`. Backdoor adjustment can only control for
confounders you named *and measured*. Unmeasured confounding is invisible to the
method, invisible to the refutation, and the single most common reason
observational causal estimates turn out wrong. The ceiling encodes that
permanently so it does not depend on anyone remembering it.

### What it is good for

A well-specified `causal_effect` is a considerably better basis for a decision
than a correlation, and considerably worse than an experiment. Treat it as the
best available answer when a randomized trial is impossible — and as a reason to
run the trial when it is possible.
