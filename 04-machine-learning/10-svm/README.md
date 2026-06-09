# Support Vector Machines

## Overview

Support Vector Machines classify examples by finding a decision boundary with a large margin. Instead of merely separating classes, an SVM tries to separate them confidently. The training objective focuses on examples near or inside the margin; these support vectors define the boundary.

SVMs are less dominant than tree ensembles and neural networks in modern production, but they remain important for understanding margins, convex classification, kernels, and robust baselines for small or medium datasets.

## Math / Derivation

For labels `y_i in {-1, 1}` and linear score `f(x) = w^T x + b`, the hard-margin SVM solves:

```
minimize   (1/2)||w||_2^2
subject to y_i(w^T x_i + b) >= 1 for all i
```

The margin width is `2 / ||w||`. Minimizing `||w||` maximizes the margin.

For nonseparable data, hinge loss gives the soft-margin objective:

```
minimize (1/2)||w||_2^2 + C sum_i max(0, 1 - y_i f(x_i))
```

Only examples with `y_i f(x_i) < 1` contribute hinge loss. The kernel trick replaces dot products with a kernel function:

```
K(x, x') = phi(x)^T phi(x')
```

This allows nonlinear boundaries without explicitly constructing high-dimensional features.

## Intuition

Logistic regression asks for probabilities. SVM asks for a wide street between classes. Points far from the boundary do not matter much once they are correctly classified with enough margin. Points on the edge of the street are support vectors, and they determine where the street lies.

The hyperparameter `C` controls tolerance for margin violations. Large `C` punishes violations heavily and can overfit. Small `C` allows more violations and yields a wider, more regularized margin.

## When & Why

Use linear SVMs for high-dimensional sparse features, such as text classification, when calibrated probabilities are not required. Use kernel SVMs for small or medium datasets with nonlinear boundaries.

Avoid kernel SVMs for very large datasets unless using approximations, because training and prediction can scale poorly with the number of support vectors. If probabilities are needed, calibrate SVM scores separately.

## Implementation

A later implementation pass should implement a linear soft-margin SVM using hinge loss and subgradient descent. It should compare SVM and logistic regression on separable and noisy synthetic data, plot margins in 2D, and validate hinge-loss subgradients away from the kink.

The implementation can describe kernels conceptually and optionally include a small RBF-kernel experiment if no extra repository assets are required.

## Cross-links

- `[[loss-functions]]` — hinge loss is a margin-based classification loss.
- `[[l1-l2-regularization]]` — the margin objective regularizes by weight norm.
- `[[logistic-regression]]` — SVMs and logistic regression are linear classifiers with different losses.
- `[[evaluation-metrics]]` — SVM thresholds and calibration affect precision/recall.
- `[[vectors-and-norms]]` — margin size depends on `||w||`.

## Resources

- Cortes and Vapnik, "Support-Vector Networks." *Machine Learning*, 1995.
- Schölkopf and Smola, *Learning with Kernels*, 2002.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, SVM chapter.
