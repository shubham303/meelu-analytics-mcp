# Clustering & dimensionality reduction

Unsupervised structure: grouping rows, and compressing columns.

## `cluster(session_key, table, n_clusters=None)`

k-means over the table's features, with labels written back as a column.

Preprocessing is handled for you: features are scaled, and categoricals are
one-hot encoded. Scaling is not optional for k-means — the algorithm minimizes
Euclidean distance, so an unscaled column measured in thousands would dominate
one measured in units, and the "clusters" would be that column alone.

**Choosing k.** Omit `n_clusters` and the engine sweeps k from 2 to 10 and keeps
the k with the maximum **silhouette score**. That is a defensible default rather
than a correct answer — silhouette rewards compact, well-separated spherical
clusters, which is exactly what k-means produces, so it is a somewhat friendly
judge of its own work. Pass an explicit `n_clusters` when the domain says what
the groups should be.

The silhouette score also feeds the result's trust level: a weak best-silhouette
means the data does not really separate, and the result says so rather than
presenting arbitrary partitions as discovered segments.

**k-means's assumptions are real.** It looks for roughly spherical, similarly
sized groups, and it will return exactly k of them whether or not the data
contains any. Always read `profile_clusters` before believing the labels.

## `profile_clusters(session_key, table)`

Describe the clusters `cluster` produced: size, numeric means, and dominant
categories per cluster.

This is the step that turns labels into meaning. A cluster is worth reporting
only if you can say what distinguishes it, and this is what tells you. Clusters
that differ on nothing interpretable are an artifact of forcing k groups onto the
data.

## `reduce_dimensions(session_key, table, method="pca", n_components=2)`

Project the table's features into `n_components` dimensions, written back as
columns.

| Method | Behaviour |
|---|---|
| `pca` (default) | Linear. Returns explained-variance ratios per component |
| `tsne` | Non-linear, neighbourhood-preserving. Perplexity adapted to n |
| `umap` | Non-linear, better global structure. Needs `uv add umap-learn` |

**Which to use.** PCA is the one to reach for when you want to *use* the output:
it is linear, deterministic, invertible in spirit, and its explained-variance
ratios tell you honestly how much you kept. Two components explaining 30% of
variance is a warning printed on the result.

t-SNE and UMAP are visualization tools. They are excellent at revealing local
structure and actively misleading if over-read: in a t-SNE plot, distances
*between* clusters carry little meaning, and cluster sizes carry none. Neither is
a good input to a downstream model.

Components are written back as real columns, so they can be plotted with
`run_sql`, or used as features — with PCA's caveats in mind.
