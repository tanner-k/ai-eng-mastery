# kNN and Naive Bayes — Solutions

## Solution 1 — Run a kNN vote

Labels are `[1, 0, 1, 1, 0]`. Class `1` appears three times and class `0` appears twice, so unweighted kNN predicts class `1`.

## Solution 2 — Show scale sensitivity

Income may vary by tens of thousands while age varies by tens. In Euclidean distance, squared income differences can dominate age differences even if age is predictive. Standardization, normalization, or a domain-specific distance metric helps put features on comparable scales.

## Solution 3 — Compute Laplace-smoothed probability

```
p("free" | spam) = (8 + 1) / (100 + 1 * 50)
                 = 9 / 150
                 = 0.06
```

## Solution 4 — Use Naive Bayes log scores

Multiplying many small probabilities can underflow to zero in floating-point arithmetic. Taking logs converts the product into a sum:

```
log product_j p_j = sum_j log p_j
```

The class with the largest product also has the largest log product because log is monotonic.

## Solution 5 — Compare model assumptions

kNN is likely better when labels follow local geometric neighborhoods, such as clustered embeddings with meaningful distances. Naive Bayes is likely better for sparse bag-of-words text where token counts provide additive class evidence and prediction latency must be low.
