# Decision Trees — Solutions

## Solution 1 — Compute Gini impurity

For counts `[6, 4]`, probabilities are `0.6` and `0.4`:

```
Gini = 1 - 0.6^2 - 0.4^2 = 0.48
```

For `[10, 0]`:

```
Gini = 1 - 1^2 - 0^2 = 0
```

## Solution 2 — Compute split gain

Parent Gini is `0.48`. Left child `[5, 1]` has:

```
1 - (5/6)^2 - (1/6)^2 = 10/36 = 0.2778
```

Right child `[1, 3]` has:

```
1 - (1/4)^2 - (3/4)^2 = 0.375
```

Weighted child impurity:

```
(6/10)*0.2778 + (4/10)*0.375 = 0.3167
```

Gain is `0.48 - 0.3167 = 0.1633`.

## Solution 3 — Regression leaf prediction

The squared-error minimizing prediction is the mean:

```
(2 + 4 + 9) / 3 = 5
```

SSE is `(2-5)^2 + (4-5)^2 + (9-5)^2 = 9 + 1 + 16 = 26`.

## Solution 4 — Diagnose overfitting

Use smaller `max_depth`, larger `min_samples_leaf`, larger `min_samples_split`, minimum impurity decrease, post-pruning, or move to an ensemble with validation tuning.

## Solution 5 — Explain interaction modeling

Example: approve a request only when `income > 100k` and `debt_ratio < 0.3`. A tree can first split on income and then split on debt ratio only inside the high-income branch. A plain linear model needs an engineered interaction or nonlinear boundary to represent that conditional rule cleanly.
