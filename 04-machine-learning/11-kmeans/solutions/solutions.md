# K-Means — Solutions

## Solution 1 — Assign points to centroids

Distances to `mu_1 = 1` and `mu_2 = 8`:

```
0: 1 vs 8 -> cluster 1
2: 1 vs 6 -> cluster 1
9: 8 vs 1 -> cluster 2
10: 9 vs 2 -> cluster 2
```

Assignments are `{0, 2}` to `mu_1` and `{9, 10}` to `mu_2`.

## Solution 2 — Update centroids

```
mu_1 = (0 + 2) / 2 = 1
mu_2 = (9 + 10) / 2 = 9.5
```

## Solution 3 — Prove the mean update

For fixed points, minimize:

```
J(mu) = sum_i ||x_i - mu||^2
```

Differentiate:

```
dJ/dmu = 2 sum_i (mu - x_i)
```

Set to zero:

```
n mu = sum_i x_i
mu = (1/n) sum_i x_i
```

So the mean is optimal.

## Solution 4 — Diagnose scale sensitivity

Euclidean distance adds squared feature differences. A dollar feature ranging in thousands can dominate a rate feature ranging from 0 to 1, even if the rate is more semantically important. Standardization, normalization, or a domain-specific distance metric helps.

## Solution 5 — Choose K

The elbow suggests `K=4` is a reasonable candidate because additional clusters provide diminishing inertia reduction. It is not proof because inertia must decrease with larger `K`, the elbow can be subjective, and real data may have continuous structure rather than discrete clusters. Validate with stability, silhouette, and downstream utility.
