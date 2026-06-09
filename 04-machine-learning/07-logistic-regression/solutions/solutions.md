# Logistic Regression — Solutions

## Solution 1 — Convert logit to probability

```
sigmoid(-2) = 1 / (1 + exp(2)) ~= 0.119
sigmoid(0) = 0.5
sigmoid(2) = 1 / (1 + exp(-2)) ~= 0.881
```

Negative logits imply probabilities below 0.5, zero maps to 0.5, and positive logits imply probabilities above 0.5.

## Solution 2 — Derive the BCE gradient

For `L = -y log p - (1-y) log(1-p)` and `p = sigmoid(z)`, first:

```
dL/dp = -y/p + (1-y)/(1-p)
dp/dz = p(1-p)
```

Then:

```
dL/dz = [-y/p + (1-y)/(1-p)] p(1-p)
       = -y(1-p) + (1-y)p
       = p - y
```

## Solution 3 — Interpret coefficients

Odds multiply by `exp(0.7) ~= 2.01`. A one-standard-deviation increase in the feature roughly doubles the odds, holding other features fixed.

## Solution 4 — Handle separable data

If classes are perfectly separable, increasing the coefficient magnitude makes predicted probabilities closer to 1 for positives and 0 for negatives, continually improving log-likelihood without a finite optimum. L1/L2 regularization, early stopping, or limiting model capacity prevents unbounded growth.

## Solution 5 — Tune a threshold

Lower the threshold below `0.5`. More examples will be predicted positive, which usually increases recall and decreases precision because more false positives are included.
