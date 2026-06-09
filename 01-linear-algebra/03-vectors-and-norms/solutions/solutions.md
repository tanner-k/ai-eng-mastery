# Vectors and Norms — Solutions

---

## Solution 1 — Compute common norms

For x = `[3, -4, 12]`:

```
||x||_1 = |3| + |-4| + |12| = 19
```

```
||x||_2 = sqrt(3^2 + (-4)^2 + 12^2)
        = sqrt(9 + 16 + 144)
        = sqrt(169)
        = 13
```

```
||x||_inf = max(3, 4, 12) = 12
```

The L2 unit vector is:

```
x / ||x||_2 = [3/13, -4/13, 12/13]
```

---

## Solution 2 — Compare dot product and cosine similarity

First compute `||q||_2 = sqrt(5)`.

For a:

```
q dot a = 1*2 + 2*4 = 10
||a||_2 = sqrt(20) = 2sqrt(5)
cos(q, a) = 10 / (sqrt(5) * 2sqrt(5)) = 1
```

For b:

```
q dot b = 1*4 + 2*1 = 6
||b||_2 = sqrt(17)
cos(q, b) = 6 / sqrt(85) ~= 0.651
```

For c:

```
q dot c = 1*(-1) + 2*(-2) = -5
||c||_2 = sqrt(5)
cos(q, c) = -5 / 5 = -1
```

The highest dot product is a with 10. The highest cosine similarity is also a with 1. The important distinction is that a wins cosine because it points exactly in the same direction as q, not because it has larger magnitude.

---

## Solution 3 — Derive gradients of norm-based objectives

The L2 norm is:

```
||x||_2 = (sum_i x_i^2)^(1/2)
```

For x != 0:

```
d/dx_j ||x||_2 = (1/2)(sum_i x_i^2)^(-1/2) * 2x_j
               = x_j / ||x||_2
```

So:

```
grad_x ||x||_2 = x / ||x||_2
```

For the squared objective:

```
(1/2)||x||_2^2 = (1/2) sum_i x_i^2
```

Therefore:

```
grad_x (1/2)||x||_2^2 = x
```

The squared objective is easier to optimize because its gradient has no division by `||x||_2`, is defined at x = 0, and is linear in x.

---

## Solution 4 — Diagnose gradient clipping

The sequence suggests exploding gradients or an unstable optimization step. The rapid growth from 3.7 to 58.0 to 420.0 is a warning sign even before the loss becomes non-finite.

With global norm clipping threshold 5.0, a gradient vector g with norm 58.0 is rescaled:

```
g_clipped = g * (5.0 / 58.0)
```

Its direction is preserved, but its norm becomes 5.0.

Global norm clipping is usually preferred because it preserves the overall gradient direction. Per-element clipping changes coordinates independently and can distort the descent direction.

---

## Solution 5 — Choose a retrieval metric

Raw dot product is:

```
q dot d = ||q|| ||d|| cos(theta)
```

So large document norms can dominate the score even when the angle to the query is not especially small. If document norm correlates with length, long documents may rank highly because of magnitude rather than semantic alignment.

Cosine similarity is the first metric to try because it normalizes by both vector lengths:

```
cos(q, d) = (q dot d) / (||q|| ||d||)
```

Before switching, check whether vector norm carries useful relevance signal. Sometimes norm reflects confidence, specificity, popularity, or text richness. Evaluate retrieval quality on labeled queries rather than assuming cosine is always better.
