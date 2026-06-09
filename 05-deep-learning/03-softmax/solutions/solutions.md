# Softmax — Solutions

## Solution 1 — Compute softmax by hand

Subtracting the max gives `[-3, -2, 0]`. Exponentials are approximately `[0.050, 0.135, 1.000]`, with sum `1.185`. Probabilities are approximately `[0.042, 0.114, 0.844]`.

## Solution 2 — Prove shift invariance

```text
softmax(z - c)_i = exp(z_i - c) / sum_j exp(z_j - c)
                 = exp(z_i) exp(-c) / (exp(-c) sum_j exp(z_j))
                 = exp(z_i) / sum_j exp(z_j)
```

The common factor cancels.

## Solution 3 — Derive the Jacobian

Let `S = sum_k exp(z_k)` and `p_i = exp(z_i)/S`.

If `i = j`:

```text
dp_i/dz_i = p_i(1 - p_i)
```

If `i != j`:

```text
dp_i/dz_j = -p_i p_j
```

Together, `J_ij = p_i(1[i = j] - p_j)`.

## Solution 4 — Diagnose numerical overflow

`exp(1000)` exceeds normal floating-point range. Subtract the maximum logit, `1002`, before exponentiation:

```text
[1000, 1001, 1002] - 1002 = [-2, -1, 0]
```

The resulting softmax is identical but safe to compute.

## Solution 5 — Temperature behavior

`T = 0.5` divides logits by a smaller number, increasing gaps and producing the sharpest distribution. `T = 1` is the baseline. `T = 2` shrinks gaps and produces the flattest distribution. As temperature rises, entropy increases.
