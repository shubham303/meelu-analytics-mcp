# Association test selection

`analyze_association(session_key, table, col_a, col_b)` answers one question —
"are these two columns related?" — and picks the correct statistical test itself.

This page documents how. It is the clearest example of what "deterministic method
selection" means in this engine, and the pattern every other routing decision
follows.

## Why the engine chooses, not you

Choosing a statistical test is a mechanical consequence of two things: the types
of the columns, and whether the parametric assumptions hold. Both are checkable
from the data. Nothing about that decision benefits from a language model's
judgment, and quite a lot about it suffers — the classic failure is running a
Pearson correlation on a categorical predictor, which produces a number that
looks fine and means nothing.

So the caller states the *question*, and the engine states the *method* in the
result. Every assumption check that drove the choice is recorded in the result's
`metadata`, so the reasoning is auditable after the fact.

## The routing table

| Column pair | Assumptions hold | Assumptions fail | Effect size reported |
|---|---|---|---|
| continuous × continuous | Pearson correlation (both normal) | Spearman rank correlation | r / ρ, plus r² |
| categorical × continuous, 2 groups | Welch's t-test | Mann-Whitney U | η² and ω² (parametric) / ε² (rank-based) |
| categorical × continuous, 3+ groups | One-way ANOVA | Kruskal-Wallis | η² and ω² / ε² |
| categorical × categorical | Chi-square | Fisher's exact (2×2, small counts) | Cramér's V |

Welch's t-test is the default two-group parametric test rather than Student's,
because it does not assume equal variances and costs almost nothing when they
are equal.

## The assumption checks

Before routing, the engine checks:

- **Normality** — of each continuous column, or of each group's values in the
  grouped cases. Failure sends continuous × continuous to Spearman and the
  grouped cases to their rank-based equivalents.
- **Equal variance** — across groups. Failure rules out one-way ANOVA in favour
  of Kruskal-Wallis.
- **Sample size** — overall and per group.
- **Expected cell counts** — for categorical × categorical. Sparse 2×2 tables go
  to Fisher's exact, where chi-square's approximation is unreliable.

Each check's outcome appears in `metadata`, so a result never asks you to take
the routing on faith.

## Effect size is always reported

A p-value tells you whether an effect is distinguishable from noise; it says
nothing about whether the effect matters. With enough rows, a trivial difference
is significant. So every test returns an effect size alongside its statistic —
r and r², η²/ω²/ε², or Cramér's V — and the summary line leads with the magnitude.

## Edge cases

- **A constant column** returns a definitive "no association" rather than a
  fabricated statistic. A column with one value cannot covary with anything, and
  the correlation formula would divide by zero.
- **Fewer than 10 usable rows** is a decline, not a result. See the
  [Honesty model](honesty-model.md).
- **Missing values** are dropped pairwise, and the usable n is reported in
  `basis`, so the trust level reflects the rows actually used rather than the
  table's length.

## Column typing drives all of this

Routing depends on knowing whether a column is continuous, nominal, or ordinal.
The engine infers types on ingest, but an unrefined "categorical" label is not
enough to route a test, and a numeric-looking identifier must never be treated as
continuous.

Type your columns before running tests — see
[Column typing](tools/column-typing.md).

## Scanning everything at once

`association_matrix(session_key, table)` runs the same routing over every
testable pair of columns and returns a matrix of effect sizes. It is the fast way
to find what is worth investigating; follow up on interesting cells with
`analyze_association` for the full statistic, assumptions, and trust block.

Treat the matrix as exploratory. Testing every pair means many tests, and some
will look significant by chance alone.
