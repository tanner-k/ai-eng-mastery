# Decision Trees

## Overview

Decision trees make predictions by recursively splitting the feature space into regions and assigning a simple prediction to each leaf. They are interpretable, handle nonlinear feature interactions, and require little preprocessing. They are also high-variance models that can overfit badly without depth, leaf-size, or pruning controls.

Trees matter in AI engineering both directly and as the base learner for `[[random-forests-xgboost]]`.

## Math / Derivation

At each node, a tree chooses a split that maximizes impurity reduction.

For classification, common impurity measures are Gini impurity:

```
Gini(S) = 1 - sum_k p_k^2
```

and entropy:

```
H(S) = -sum_k p_k log p_k
```

For a split into left and right subsets:

```
gain = impurity(parent)
       - (n_left/n) impurity(left)
       - (n_right/n) impurity(right)
```

For regression, impurity is often squared error around the node mean:

```
SSE(S) = sum_{i in S} (y_i - ybar_S)^2
```

The leaf prediction is the majority class for classification or mean target for regression.

## Intuition

A tree asks a sequence of yes/no questions. Each question should make the remaining examples more homogeneous. A shallow tree captures broad structure. A deep tree can memorize individual training examples.

Trees naturally model interactions: a split on feature B can matter only after a split on feature A. That makes them strong on tabular data where feature interactions are important and hard to specify manually.

## When & Why

Use decision trees when interpretability, nonlinear rules, mixed feature types, or missing-value handling are important. Use them as baselines for tabular problems and as a conceptual foundation for ensembles.

Control overfitting with maximum depth, minimum samples per leaf, minimum impurity decrease, pruning, and validation metrics. Single trees are unstable: small data changes can produce different split structures. Ensembles reduce this variance.

## Implementation

A later implementation pass should implement a small CART-style tree from scratch for classification and regression. It should search numeric thresholds, compute Gini or SSE gain, stop by depth and leaf size, and validate predictions on synthetic datasets.

The implementation should emphasize correctness over speed first, then discuss vectorized split search as a future improvement.

## Cross-links

- `[[evaluation-metrics]]` — split criteria differ from final validation metrics.
- `[[bias-variance]]` — deep trees have low bias and high variance.
- `[[random-forests-xgboost]]` — ensembles stabilize or boost trees.
- `[[loss-functions]]` — regression trees usually minimize squared error in leaves.
- `[[knn-naive-bayes]]` — trees are another nonparametric-style baseline for tabular data.

## Resources

- Breiman, Friedman, Olshen, and Stone, *Classification and Regression Trees*, 1984.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, Chapter 9.
- scikit-learn User Guide, "Decision Trees."
