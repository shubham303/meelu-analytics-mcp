"""Dimensionality reduction family.

reduce_dimensions projects the feature matrix to a few components and writes them
back as columns (so clustering on the reduced space becomes an ordinary query).
PCA and t-SNE are scikit-learn native; UMAP is an optional lazy import.
"""
from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ....shared import honesty
from ....shared.identity import _lazy_import
from ....shared.results import Result
from .. import _prep

_RANDOM_STATE = 0


def reduce_dimensions(
    store: Any,
    method: str = "pca",
    n_components: int = 2,
) -> Result:
    """Reduce table dimensionality and write the components back as columns.

    Components are written back as ``<method>_0 .. <method>_{n-1}`` so downstream
    analysis on the reduced space is just another query.

    Args:
        store: The Store instance holding the table.
        method: One of "pca", "tsne", "umap".
        n_components: Number of output dimensions (typically 2 for visualization).

    Returns:
        Result with explained variance (PCA) and the column names written back.
    """
    method = method.lower()
    # Exclude columns we previously wrote back so reduction runs on real features.
    exclude = tuple(
        c for c in store._table.schema()
        if c == "cluster" or c.startswith(("pca_", "tsne_", "umap_"))
    )
    X, frame, features = _prep.numeric_matrix(store, exclude=exclude, scale=True)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=_RANDOM_STATE)
        coords = reducer.fit_transform(X)
        explained = [float(v) for v in reducer.explained_variance_ratio_]
    elif method == "tsne":
        perplexity = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
        reducer = TSNE(
            n_components=n_components, random_state=_RANDOM_STATE, perplexity=perplexity
        )
        coords = reducer.fit_transform(X)
        explained = None
    elif method == "umap":
        umap = _lazy_import("umap")  # optional dependency
        reducer = umap.UMAP(n_components=n_components, random_state=_RANDOM_STATE)
        coords = reducer.fit_transform(X)
        explained = None
    else:
        raise ValueError(f"Unknown method {method!r}; expected pca, tsne, or umap.")

    written = []
    for i in range(n_components):
        col = f"{method}_{i}"
        # Components are usable features (e.g. reduce_dimensions → cluster), so
        # they stay eligible in feature matrices rather than being marked derived.
        store.write_back_column(col, [float(v) for v in coords[:, i]], feature=True)
        written.append(col)

    # Honesty seam: confidence from sample size, plus the mandatory distortion caveat.
    # When variance-explained is available (PCA), fold it in — a low-variance
    # projection keeps little of the original signal.
    trust = honesty.from_sample_size(int(X.shape[0]), low=30, moderate=200, label="rows")
    trust = honesty.with_caveats(
        trust,
        "Projecting to fewer dimensions distorts distances; treat the layout as approximate.",
    )
    if explained is not None:
        kept = sum(explained)
        trust = honesty.with_caveats(
            trust, f"The projection keeps only {kept:.0%} of the variance."
        )
        if kept < 0.5:
            trust = honesty.Trust(
                level=honesty.TrustLevel.LOW, caveats=trust.caveats,
                basis=[*trust.basis, f"variance_explained={kept:.3f}"],
            )

    return Result(
        method=method,
        summary=(
            f"Reduced {len(features)} features to {n_components}D via {method.upper()}"
            + (f" ({sum(explained):.0%} variance)" if explained else "")
        ),
        values={
            "n_components": n_components,
            "explained_variance_ratio": explained,
            "columns": written,
        },
        metadata={"method": method, "features": features, "columns_written": written},
        trust=trust,
    )
