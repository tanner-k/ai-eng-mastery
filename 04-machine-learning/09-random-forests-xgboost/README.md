# Random Forests and XGBoost

## Overview

Random forests and gradient-boosted trees are the dominant classical machine learning methods for tabular data. Both build ensembles of decision trees, but they use different strategies.

Random forests average many decorrelated trees to reduce variance. Gradient boosting builds trees sequentially, where each new tree corrects errors from the current ensemble. XGBoost is a highly optimized, regularized gradient boosting system.

## Math / Derivation

Random forest prediction averages `B` trees:

```
f(x) = (1/B) sum_b T_b(x)
```

Each tree is trained on a bootstrap sample, and each split considers only a random subset of features. Bootstrapping and feature subsampling reduce correlation among trees, making averaging more effective.

Gradient boosting builds an additive model:

```
F_M(x) = sum_{m=1}^M eta f_m(x)
```

At each step, the next tree is fit to the negative gradient of the loss with respect to current predictions:

```
r_i_m = - d L(y_i, F_{m-1}(x_i)) / d F_{m-1}(x_i)
```

For squared error, these pseudo-residuals are ordinary residuals. XGBoost adds regularization on tree complexity and uses second-order loss approximations to score splits.

## Intuition

A single deep tree is noisy. A random forest asks many noisy trees to vote, and the noise cancels out when the trees make different mistakes.

Boosting is more like staged repair. Start with a weak model, inspect what it gets wrong, fit a small tree to those errors, and repeat. The learning rate controls how much each repair step is trusted.

## When & Why

Use random forests for strong, low-maintenance tabular baselines, robustness, and quick feature importance checks. They are less sensitive to hyperparameters than boosted trees but can be larger and less accurate on structured tabular benchmarks.

Use XGBoost or similar gradient boosting when tabular performance matters and careful validation is available. Tune learning rate, number of trees, max depth, subsampling, column sampling, and regularization. Boosting can overfit if too many trees are added or if leakage is present.

## Implementation

A later implementation pass should build a small random forest from the decision-tree implementation and a simple gradient boosting regressor for squared error. It should compare single tree, forest, and boosting on the same synthetic dataset and report train/validation metrics.

The implementation should describe XGBoost concepts but does not need to recreate the full XGBoost system in the first implementation pass.

## Cross-links

- `[[decision-trees]]` — forests and boosted trees use trees as base learners.
- `[[bias-variance]]` — bagging reduces variance; boosting can reduce bias and variance.
- `[[evaluation-metrics]]` — boosted models need validation and early stopping.
- `[[loss-functions]]` — boosting fits gradients of the chosen loss.
- `[[l1-l2-regularization]]` — XGBoost regularizes leaf weights and tree complexity.

## Resources

- Leo Breiman, "Random Forests." *Machine Learning*, 2001.
- Jerome Friedman, "Greedy Function Approximation: A Gradient Boosting Machine." 2001.
- Tianqi Chen and Carlos Guestrin, "XGBoost: A Scalable Tree Boosting System." KDD 2016.
