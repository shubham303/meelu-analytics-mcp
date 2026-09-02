"""Market-basket / association-rule mining — the cross-sell insight primitive.

``market_basket`` answers "customers who buy X also buy Y". Given a transaction id
column and an item column (the long/tidy shape of an order-lines table), it mines
frequent itemsets and association rules.

Library: **mlxtend** (``apriori`` + ``association_rules``), imported lazily via the
optional ``insights`` extra. We only write the thin glue that turns the tidy table
into the one-hot basket matrix mlxtend expects and formats the rules as plain dicts.
"""
from __future__ import annotations

from typing import Any

from ....shared import honesty
from ....shared.results import Result


def market_basket(
    store: Any,
    transaction_column: str,
    item_column: str,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    max_rules: int = 50,
) -> Result:
    """Mine association rules from an order-lines table.

    Args:
        store: The Store/Table instance.
        transaction_column: Column identifying a basket/order (rows grouped by it).
        item_column: Column naming the item in each row.
        min_support: Minimum itemset support for apriori (fraction of baskets).
        min_confidence: Minimum rule confidence to keep.
        max_rules: Cap on returned rules (ranked by lift).

    Returns:
        Result with ``rules`` (antecedents → consequents with support/confidence/lift),
        ordered by lift descending.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "market_basket needs mlxtend. Install it with:  pip install 'tabint[insights]'."
        ) from exc

    frame = store.get_frame()
    for col in (transaction_column, item_column):
        if col not in frame.columns:
            raise ValueError(f"Column {col!r} not in table.")

    baskets = (
        frame[[transaction_column, item_column]]
        .dropna()
        .groupby(transaction_column)[item_column]
        .apply(lambda s: sorted(set(s.astype(str))))
        .tolist()
    )
    if not baskets:
        raise ValueError("No non-empty transactions to mine.")

    encoder = TransactionEncoder()
    onehot = encoder.fit_transform(baskets)
    import pandas as pd

    basket_df = pd.DataFrame(onehot, columns=encoder.columns_)

    n_transactions = len(baskets)

    itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        trust = honesty.decline(
            f"No itemsets met min_support={min_support} — there aren't enough baskets that "
            "share the same items to find any reliable pattern"
            + (f" (only {n_transactions} transactions)." if n_transactions < 100 else "."),
            caveats=[
                "Association rules need enough transactions/baskets to be reliable.",
                "Try lowering min_support, but patterns from very few baskets can be spurious.",
            ],
            basis=[f"n_transactions={n_transactions}"],
        )
        return Result(
            method="apriori_association_rules_declined",
            summary=f"Declined: no itemsets met min_support={min_support} — too little co-occurrence to mine.",
            values={},
            metadata={"transaction_column": transaction_column, "item_column": item_column,
                      "n_transactions": n_transactions, "min_support": min_support},
            trust=trust,
        )

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False).head(max_rules)

    formatted = [
        {
            "antecedents": sorted(r["antecedents"]),
            "consequents": sorted(r["consequents"]),
            "support": float(r["support"]),
            "confidence": float(r["confidence"]),
            "lift": float(r["lift"]),
        }
        for _, r in rules.iterrows()
    ]
    top = formatted[0] if formatted else None
    summary = (
        f"{' + '.join(top['antecedents'])} → {' + '.join(top['consequents'])} "
        f"(lift {top['lift']:.2f})"
        if top
        else f"No rules met min_confidence={min_confidence}"
    )

    trust = honesty.from_sample_size(n_transactions, low=100, moderate=500, label="transactions")
    trust = honesty.with_caveats(
        trust,
        "Association rules are correlational, not causal — buying X doesn't cause buying Y.",
        "Lift/confidence on rare item pairs can be spurious — check the support before acting.",
        "Rules need enough transactions/baskets to be reliable; sparse baskets give unstable numbers.",
    )

    return Result(
        method="apriori_association_rules",
        summary=summary,
        values={"rules": formatted},
        metadata={
            "transaction_column": transaction_column,
            "item_column": item_column,
            "n_transactions": n_transactions,
            "min_support": min_support,
            "min_confidence": min_confidence,
            "n_rules": len(formatted),
        },
        trust=trust,
    )
