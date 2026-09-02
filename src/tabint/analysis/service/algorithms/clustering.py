"""Clustering family — k-means with automatic k selection.

cluster scales features, fits k-means, and (when k is not given) picks k by the
silhouette method. Labels are written back as a column so follow-up questions
become ordinary queries. profile_clusters then characterises each cluster.
Library: scikit-learn.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ....shared import honesty
from ....shared.results import Result
from ..validation.dtypes import classify_column
from .. import _prep

_LABEL_COLUMN = "cluster"
_K_MIN, _K_MAX = 2, 10

_CLUSTER_CAVEATS = (
    "The number of clusters k is a modeling choice, not ground truth.",
    "Clusters are descriptive groupings — different settings can give different groups.",
)


def cluster(store: Any, n_clusters: int | None = None) -> Result:
    """Cluster rows with k-means and write the labels back as a column.

    Features are standard-scaled and one-hot encoded first. If ``n_clusters`` is
    None, k in [2, 10] is chosen by maximising the silhouette score.

    Args:
        store: The Store instance holding the table.
        n_clusters: Number of clusters, or None to select automatically.

    Returns:
        Result with the chosen k, silhouette score, cluster sizes, and label column.
    """
    X, frame, features = _prep.numeric_matrix(store, exclude=(_LABEL_COLUMN,), scale=True)
    n_samples = X.shape[0]

    # Degenerate: fewer than 2 rows can't be partitioned — one trivial cluster.
    if n_samples < 2:
        store.write_back_column(_LABEL_COLUMN, [0] * n_samples)
        reason = f"Only {n_samples} row(s) — too few to partition into clusters."
        return Result(
            method="kmeans_declined",
            summary=f"Declined: {reason}",
            values={"n_clusters": 1, "silhouette": None, "cluster_sizes": {0: n_samples}},
            metadata={"algorithm": "kmeans", "k_selection": "degenerate",
                      "features": features, "label_column": _LABEL_COLUMN},
            trust=honesty.decline(reason, basis=[f"n={n_samples}"]),
        )

    if n_clusters is not None:
        if n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={n_clusters} exceeds the number of rows ({n_samples})."
            )
        best_k, best_labels = n_clusters, _fit(X, n_clusters)
        best_sil = _safe_silhouette(X, best_labels)
    else:
        best_k, best_labels, best_sil = _choose_k(X, n_samples)

    store.write_back_column(_LABEL_COLUMN, [int(v) for v in best_labels])
    sizes = pd.Series(best_labels).value_counts().sort_index()

    # Honesty seam: base confidence from sample size, then modulate by how well the
    # clusters actually separate (silhouette). Weak separation caps trust at LOW.
    trust = honesty.from_sample_size(n_samples, low=30, moderate=200, label="rows")
    trust = honesty.with_caveats(trust, *_CLUSTER_CAVEATS)
    if best_sil is not None and best_sil < 0.25:
        trust = honesty.Trust(
            level=honesty.TrustLevel.LOW,
            caveats=honesty.with_caveats(
                trust, "The clusters are weakly separated (low silhouette) — the grouping "
                "may not reflect real structure.").caveats,
            basis=[*trust.basis, f"silhouette={best_sil:.3f}"],
        )

    return Result(
        method="kmeans",
        summary=(
            f"{best_k} clusters"
            + (f", silhouette={best_sil:.3f}" if best_sil is not None else "")
        ),
        values={
            "n_clusters": int(best_k),
            "silhouette": None if best_sil is None else float(best_sil),
            "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
        },
        metadata={
            "algorithm": "kmeans",
            "k_selection": "fixed" if n_clusters is not None else "silhouette",
            "features": features,
            "label_column": _LABEL_COLUMN,
        },
        trust=trust,
    )


def profile_clusters(store: Any) -> Result:
    """Characterise each cluster: size, numeric means, dominant categorical values.

    Requires cluster() to have been run first (the ``cluster`` label column must
    exist in the store).

    Args:
        store: The Store instance holding the table (with a cluster label column).

    Returns:
        Result mapping each cluster id to its distinguishing characteristics.
    """
    frame = store.get_frame()
    if _LABEL_COLUMN not in frame.columns:
        raise ValueError("No cluster labels found — call cluster() first.")

    numeric, nominal, ordinal = _prep.feature_columns(store, exclude=(_LABEL_COLUMN,))
    categorical = nominal + ordinal
    profiles: dict[int, Any] = {}

    for cid, group in frame.groupby(_LABEL_COLUMN, observed=True):
        entry: dict[str, Any] = {"size": int(group.shape[0])}
        entry["numeric_means"] = {
            c: float(pd.to_numeric(group[c], errors="coerce").mean()) for c in numeric
        }
        entry["dominant_categories"] = {
            c: (str(group[c].mode().iloc[0]) if not group[c].mode().empty else None)
            for c in categorical
        }
        profiles[int(cid)] = entry

    trust = honesty.from_sample_size(
        int(frame.shape[0]), low=30, moderate=200, label="rows"
    )
    trust = honesty.with_caveats(
        trust,
        "This just summarizes the existing cluster groups (sizes, means, dominant "
        "categories) — it describes them, it doesn't validate that the clusters are real.",
    )

    return Result(
        method="profile_clusters",
        summary=f"Characterised {len(profiles)} clusters",
        values={"clusters": profiles},
        metadata={"numeric_features": numeric, "categorical_features": categorical},
        trust=trust,
    )


def _fit(X: Any, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


def _safe_silhouette(X: Any, labels: np.ndarray) -> float | None:
    # silhouette needs 2 <= n_labels <= n_samples - 1.
    n_labels = len(set(labels))
    if 2 <= n_labels <= X.shape[0] - 1:
        return float(silhouette_score(X, labels))
    return None


def _choose_k(X: Any, n_samples: int) -> tuple[int, np.ndarray, float | None]:
    """Pick k in [2, min(_K_MAX, n_samples-1)] by best silhouette.

    Returns silhouette=None when no valid k could be scored (e.g. n_samples == 2),
    rather than fabricating a placeholder value.
    """
    k_max = min(_K_MAX, n_samples - 1)
    best_k, best_labels, best_sil = 2, _fit(X, 2), None
    for k in range(_K_MIN, k_max + 1):
        labels = _fit(X, k)
        sil = _safe_silhouette(X, labels)
        if sil is not None and (best_sil is None or sil > best_sil):
            best_k, best_labels, best_sil = k, labels, sil
    return best_k, best_labels, best_sil
