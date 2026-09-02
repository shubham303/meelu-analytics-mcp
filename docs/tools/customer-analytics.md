# Customer analytics

Three standard commercial analyses, implemented deterministically.

## `market_basket(session_key, table, transaction_column, item_column, min_support=0.01, min_confidence=0.2, max_rules=50)`

Association-rule mining — "customers who buy X also buy Y" — via Apriori frequent
itemsets plus association rules (`mlxtend`, needs the
[`insights` extra](../configuration.md#dependencies)). Rules are ranked by
**lift** and capped at `max_rules`.

Input is transaction-shaped: one row per item per transaction, with
`transaction_column` identifying the basket and `item_column` the product.

**The three numbers, and why lift leads:**

| Metric | Meaning |
|---|---|
| Support | How often the itemset appears at all |
| Confidence | Given X, how often Y follows |
| Lift | How much more often than chance — 1.0 means independent |

Confidence alone is a trap. If 80% of all baskets contain bananas, then "buys
socks → buys bananas" has 80% confidence and means nothing; lift is 1.0 and says
so immediately. Ranking by lift is what keeps the popular-item rules out.

**Tuning the thresholds.** `min_support` is the real control. Lowering it finds
rarer combinations and makes the search dramatically more expensive — Apriori's
cost grows fast as the frequent-itemset space opens up. Very low support also
produces rules with high lift resting on a handful of baskets. Start at the
default and lower it deliberately.

## `rfm(session_key, table, customer_column, date_column, monetary_column)`

Recency / Frequency / Monetary quintile scoring, mapped to the canonical
segments: **Champions**, **Loyal**, **Potential Loyalist**, **At Risk**,
**Hibernating**.

Each customer is scored 1–5 on each dimension by quintile, and the segment
follows deterministically from the score combination.

Because scoring is by quintile, it is **relative to your customer base**, always.
There will be a top quintile whether your business is thriving or collapsing, and
"Champions" means "best fifth of these customers", not "good customers" in any
absolute sense. Comparing segment sizes across time periods is meaningless for
the same reason — the boundaries move with the data.

RFM's real value is as a triage: it identifies who to look at, using only three
columns almost every transactional dataset already has.

## `retention_cohorts(session_key, table, customer_column, date_column)`

A monthly retention matrix: first-purchase cohort × months since first purchase,
as both counts and rates.

Each customer joins the cohort of their first purchase month, and each cell holds
how many of that cohort were active n months later. Reading it:

- **Down a column** — is retention at month n improving for newer cohorts? This
  is the product-improvement question, and it is the one the matrix answers best.
- **Across a row** — how does a single cohort decay over its lifetime?

Two cautions worth stating. The most recent cohorts have had the fewest months to
be observed, so their rows are short and their apparent retention is not
comparable to older cohorts' — the triangular shape of the matrix is the reminder.
And month boundaries are arbitrary: a customer who buys on the 31st and again on
the 1st looks retained, while one who buys on the 1st and again 40 days later does
not.
