# L1 and L2 Regularization — Solutions

## Solution 1 — Derive the L2 gradient

Since `||w||_2^2 = sum_j w_j^2`, its gradient is `2w`. Therefore:

```
grad J(w) = grad L(w) + 2 lambda w
```

Some libraries define the penalty as `(lambda/2)||w||^2`, in which case the added gradient is `lambda w`.

## Solution 2 — Identify L1 subgradients

For `w = [-2, 0, 3]`, one valid subgradient is:

```
[-1, 0, 1]
```

The middle component can be any value in `[-1, 1]` because `|w|` has a corner at zero, so there is no unique tangent slope.

## Solution 3 — Compare penalty values

For `w_a = [3, 0, 0]`, L1 is `3` and squared L2 is `9`. For `w_b = [1, 1, 1]`, L1 is `3` and squared L2 is `3`. L1 is indifferent between these two vectors by penalty value, while squared L2 prefers the spread-out vector.

## Solution 4 — Choose a regularizer

L1 or Elastic Net is the right starting point. Pure L1 can produce a compact influential-feature list, which matches the product requirement. Elastic Net is safer if text features are correlated because the L2 component stabilizes coefficient selection among related terms.

## Solution 5 — Diagnose over-regularization

This is likely underfitting caused by excessive regularization. Reduce `lambda`, tune it on a log scale, and compare against an unregularized or weakly regularized baseline. Also verify that features are standardized so the penalty is not accidentally too strong for some columns.
