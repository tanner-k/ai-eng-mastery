# Principal Component Analysis — Solutions

## Solution 1 — Center a dataset

Column means are:

```
[(1+3+5)/3, (2+4+6)/3] = [3, 4]
```

Centered rows:

```
(-2, -2)
( 0,  0)
( 2,  2)
```

## Solution 2 — Interpret eigenvalues

Total variance is `9 + 3 + 0 = 12`.

```
component 1: 9/12 = 0.75
component 2: 3/12 = 0.25
component 3: 0/12 = 0
```

## Solution 3 — Derive the first component objective

The projected values are `Xw`. Their average squared magnitude for centered data is:

```
(1/n) ||Xw||^2 = (1/n) w^T X^T X w = w^T Sigma w
```

So maximizing projected variance subject to `||w|| = 1` is maximizing `w^T Sigma w`.

## Solution 4 — Explain reconstruction

Adding components expands the subspace available for reconstruction. The best `k+1`-dimensional PCA subspace can always choose to include the best `k`-dimensional solution plus one more direction, so squared reconstruction error cannot increase and usually decreases.

## Solution 5 — Prevent leakage

Fitting PCA on the full dataset lets validation data influence the mean and component directions. This makes validation performance too optimistic. Correct workflow: split data first, fit mean/components on training data only, then transform validation/test data with those training-fitted quantities.
