# L1 and L2 Regularization

## Overview

Regularization deliberately constrains a model so it generalizes better. L1 and L2 penalties are the two most common parameter penalties: L1 encourages sparsity, while L2 encourages small, distributed weights. Both reduce overfitting by making overly complex explanations more expensive.

For AI engineers, regularization is a control surface for `[[bias-variance]]`. Too little regularization lets the model memorize noise. Too much regularization suppresses useful signal and causes underfitting.

## Math / Derivation

Empirical risk with a penalty has the form:

```
J(theta) = L(theta) + lambda * Omega(theta)
```

For weights `w`:

```
L1: Omega(w) = ||w||_1 = sum_j |w_j|
L2: Omega(w) = ||w||_2^2 = sum_j w_j^2
```

The L2 gradient is smooth:

```
grad_w [L(w) + lambda ||w||_2^2] = grad_w L(w) + 2 lambda w
```

The L1 penalty is non-smooth at zero. Its subgradient is:

```
d|w_j|/dw_j = sign(w_j) for w_j != 0
d|w_j|/dw_j in [-1, 1] for w_j = 0
```

This kink at zero is what makes exact zeros possible. L2 shrinks weights continuously but usually does not set them exactly to zero.

## Intuition

L2 says every weight should pay rent proportional to its size. It prefers many modest weights over a few huge weights and produces smoother models.

L1 says every nonzero weight pays a fixed entry fee. If a feature is not clearly useful, the cheapest solution is to drive its coefficient exactly to zero. This makes L1 useful for feature selection and interpretability.

Geometrically, L2 uses round constraint contours, while L1 uses diamond-shaped contours with corners on the axes. Optimization often lands on those corners, creating sparse solutions.

## When & Why

Use L2 when many features may contribute weak signal, when multicollinearity makes coefficients unstable, or when you want a stable default. Weight decay in neural networks is the deep-learning version of this idea, though decoupled weight decay differs from adding an L2 term under adaptive optimizers.

Use L1 when sparsity is valuable: high-dimensional tabular data, feature selection, compressed linear models, or explanations where nonzero coefficients should be rare.

Elastic Net combines both:

```
lambda1 ||w||_1 + lambda2 ||w||_2^2
```

It is useful when correlated features make pure L1 unstable.

## Implementation

A later implementation pass should train linear and logistic models with optional L1, L2, and Elastic Net penalties. It should validate analytic gradients for L2 against `torch.autograd`, demonstrate subgradient behavior for L1, and compare coefficient sparsity and validation performance across penalty strengths.

The implementation should clearly distinguish L2 regularization added to the loss from decoupled weight decay used by optimizers such as AdamW.

## Cross-links

- `[[loss-functions]]` — penalties are added to the training objective.
- `[[gradient-descent]]` — regularization changes the gradient update.
- `[[adam-optimization]]` — adaptive optimizers make L2 and weight decay behave differently.
- `[[linear-regression]]` — ridge and lasso are regularized linear regression.
- `[[bias-variance]]` — regularization trades variance for bias.

## Resources

- Robert Tibshirani, "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society*, 1996.
- Hoerl and Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 1970.
- Zou and Hastie, "Regularization and Variable Selection via the Elastic Net." 2005.
