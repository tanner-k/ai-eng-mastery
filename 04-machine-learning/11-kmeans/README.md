# K-Means

## Overview

K-means clusters unlabeled data by assigning each point to the nearest centroid and updating centroids to the mean of assigned points. It is a simple unsupervised learning algorithm used for segmentation, vector quantization, prototype discovery, and initialization.

Its simplicity is also its limitation: k-means assumes roughly spherical clusters of similar scale under Euclidean distance.

## Math / Derivation

Given points `x_i in R^d` and `K` clusters with centroids `mu_k`, k-means minimizes within-cluster sum of squares:

```
J = sum_i ||x_i - mu_{c_i}||_2^2
```

where `c_i` is the cluster assignment for point `i`.

The standard algorithm alternates:

1. Assignment step:

```
c_i = argmin_k ||x_i - mu_k||_2^2
```

2. Update step:

```
mu_k = mean({x_i : c_i = k})
```

The update step is optimal for fixed assignments because the mean minimizes squared distance within a cluster.

## Intuition

K-means places `K` representatives in the data space. Each point joins the nearest representative. Then each representative moves to the center of its assigned points. Repeating this shrinks the total squared distance until assignments stop changing or improvement becomes tiny.

The algorithm is coordinate descent over assignments and centroids. It decreases the objective each iteration but can converge to a local optimum.

## When & Why

Use k-means when you need a fast clustering baseline, customer or embedding segmentation, prototype compression, or initialization for other algorithms. Standardize features first; Euclidean distance is scale-sensitive.

Choose `K` with domain knowledge, elbow plots, silhouette scores, stability checks, or downstream utility. Avoid blindly treating clusters as ground truth. K-means can fail on non-spherical clusters, varying densities, outliers, and categorical features without suitable encodings or distance choices.

## Implementation

A later implementation pass should implement k-means from scratch with tensorized distance computation, random and k-means++ initialization, empty-cluster handling, and convergence checks. It should compare inertia across `K` values and validate behavior on synthetic blobs and non-spherical data.

The implementation should explicitly show how feature scaling changes assignments.

## Cross-links

- `[[vectors-and-norms]]` — Euclidean distance defines assignments.
- `[[broadcasting]]` — efficient distance matrices use broadcasting.
- `[[evaluation-metrics]]` — clustering quality needs internal and downstream metrics.
- `[[pca]]` — PCA can visualize or denoise data before clustering.
- `[[bias-variance]]` — choosing `K` controls underfitting versus overfragmentation.

## Resources

- MacQueen, "Some methods for classification and analysis of multivariate observations." 1967.
- Arthur and Vassilvitskii, "k-means++: The Advantages of Careful Seeding." 2007.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, clustering sections.
