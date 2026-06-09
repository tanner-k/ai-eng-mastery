# Linear Regression

## Overview

Linear regression models a continuous target as a weighted sum of input features. It is simple, interpretable, and still one of the most important baselines in AI engineering. Many complex systems reduce to linear regression locally: final prediction heads, calibration layers, control variates, and diagnostic baselines all use the same geometry.

The model is:

```
yhat = Xw + b
```

Training usually minimizes mean squared error, connecting linear regression directly to `[[loss-functions]]`, `[[gradient-descent]]`, and Gaussian maximum likelihood.

## Math / Derivation

For `X in R^{n x d}`, `w in R^d`, and `y in R^n`:

```
L(w, b) = (1/n) ||Xw + b1 - y||_2^2
```

If the bias is folded into `X` by adding a column of ones, the ordinary least squares objective is:

```
L(beta) = (1/n) ||X beta - y||_2^2
```

Setting the gradient to zero gives the normal equations:

```
X^T X beta = X^T y
```

When `X^T X` is invertible:

```
beta = (X^T X)^{-1} X^T y
```

With L2 regularization (ridge):

```
beta = (X^T X + lambda I)^{-1} X^T y
```

The gradient for iterative training is:

```
grad_beta L = (2/n) X^T (X beta - y)
```

## Intuition

Linear regression projects the target vector onto the column space of the feature matrix. The prediction is the closest vector to `y` that can be expressed as a linear combination of the features. Residuals are what remains orthogonal to that fitted subspace.

Each coefficient is the estimated change in target for a one-unit feature change, holding other features fixed. That interpretation is only reliable when features are well-defined, not severely collinear, and the data collection process supports the comparison.

## When & Why

Use linear regression as a baseline for continuous targets, for interpretable tabular models, and when data is limited. Use ridge when features are correlated or the system is underdetermined. Use lasso or Elastic Net when sparsity matters.

Linear regression can fail when relationships are strongly nonlinear, residual variance changes with feature values, targets have heavy tails, or important interactions are missing. Residual plots and validation metrics are essential diagnostics.

## Implementation

A later implementation pass should implement linear regression two ways: closed-form least squares with numerically stable linear solves, and gradient-based training with PyTorch tensors. It should validate analytic gradients against `torch.autograd`, compare OLS and ridge on collinear features, and report MAE/RMSE/R^2 on held-out data.

The implementation should avoid explicitly inverting matrices when solving normal equations; stable solves or decompositions should be preferred.

## Cross-links

- `[[matrix-multiplication]]` — linear prediction is a matrix-vector product.
- `[[vectors-and-norms]]` — MSE is a squared norm of residuals.
- `[[loss-functions]]` — MSE defines the usual objective.
- `[[l1-l2-regularization]]` — ridge and lasso regularize linear regression.
- `[[evaluation-metrics]]` — regression quality needs held-out metrics and residual diagnostics.

## Resources

- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, Chapter 3.
- Kevin P. Murphy, *Machine Learning: A Probabilistic Perspective*, linear regression chapter.
- Gilbert Strang, *Linear Algebra and Learning from Data*, least squares chapters.
