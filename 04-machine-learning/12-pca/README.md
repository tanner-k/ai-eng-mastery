# Principal Component Analysis

## Overview

Principal Component Analysis (PCA) finds orthogonal directions of maximum variance in a dataset. It is used for dimensionality reduction, denoising, visualization, compression, and understanding feature covariance structure.

PCA is unsupervised: it uses only `X`, not labels. That makes it useful for exploration, but also means high-variance directions are not guaranteed to be predictive directions.

## Math / Derivation

Given centered data matrix `X in R^{n x d}`, the empirical covariance matrix is:

```
Sigma = (1/n) X^T X
```

The first principal component solves:

```
maximize_w w^T Sigma w
subject to ||w||_2 = 1
```

The solution is the eigenvector of `Sigma` with the largest eigenvalue. Subsequent components are orthogonal eigenvectors ordered by decreasing eigenvalue.

Equivalently, using SVD:

```
X = U S V^T
```

The rows or columns of `V^T` give principal directions, and singular values determine explained variance:

```
explained_variance_j = S_j^2 / n
explained_variance_ratio_j = S_j^2 / sum_k S_k^2
```

Projecting to `k` components:

```
Z = X V_k
X_reconstructed = Z V_k^T
```

## Intuition

PCA rotates the coordinate system so the first axis points along the widest spread of the data, the second axis points along the widest remaining orthogonal spread, and so on. Keeping the first few axes preserves as much squared reconstruction energy as possible among all linear `k`-dimensional projections.

If two features are strongly correlated, PCA can replace them with a smaller set of uncorrelated components.

## When & Why

Use PCA for visualization, compression, denoising, whitening, collinearity diagnostics, and preprocessing when many features are redundant. Always fit PCA only on training data, then apply the learned transform to validation/test data to avoid leakage.

Avoid PCA when component interpretability is critical, when nonlinear structure dominates, or when low-variance features carry the predictive signal. For supervised tasks, validate whether PCA helps the downstream metric rather than assuming variance preservation implies better prediction.

## Implementation

A later implementation pass should implement PCA from centered tensors using both covariance eigendecomposition and SVD. It should compare explained variance, reconstruction error, whitening, and downstream classification/regression performance with different component counts.

The implementation should demonstrate train-only fitting and validation transformation to avoid data leakage.

## Cross-links

- `[[eigendecomposition-svd]]` — PCA is built from eigenvectors or singular vectors.
- `[[vectors-and-norms]]` — reconstruction error is a squared norm.
- `[[matrix-multiplication]]` — projections and reconstructions are matrix products.
- `[[kmeans]]` — PCA is often used to visualize clusters.
- `[[evaluation-metrics]]` — component count should be selected by validation behavior.

## Resources

- Jolliffe and Cadima, "Principal component analysis: a review and recent developments." 2016.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, PCA sections.
- Bishop, *Pattern Recognition and Machine Learning*, dimensionality reduction sections.
