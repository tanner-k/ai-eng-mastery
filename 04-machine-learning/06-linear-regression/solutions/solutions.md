# Linear Regression — Solutions

## Solution 1 — Derive the normal equations

Expand the objective:

```
L(beta) = (X beta - y)^T (X beta - y)
```

The gradient is:

```
grad_beta L = 2 X^T (X beta - y)
```

Set it to zero:

```
X^T (X beta - y) = 0
X^T X beta = X^T y
```

## Solution 2 — Compute a one-feature fit

The points lie exactly on `y = 2x + 1`. Therefore `w = 2` and `b = 1`, with zero residuals.

## Solution 3 — Identify collinearity

Nearly identical features make `X^T X` ill-conditioned. Small data perturbations can cause large coefficient changes because many coefficient combinations make similar predictions. Ridge adds `lambda I`, improving conditioning and shrinking coefficients toward more stable values.

## Solution 4 — Interpret residual diagnostics

Increasing residual spread suggests heteroscedasticity: error variance is not constant. Possible responses include transforming the target, modeling variance, using weighted least squares, using robust losses, or adding features/interactions that explain the changing variance.

## Solution 5 — Compare OLS and gradient descent

Prefer closed-form or linear-solve methods for small to medium dense problems when exact least-squares fitting is cheap and stable. Prefer gradient descent for very large datasets, streaming or mini-batch settings, custom losses, regularizers not handled by closed form, or models that are only partly linear.
