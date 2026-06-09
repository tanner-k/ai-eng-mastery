# Cross-Entropy and NLL Loss — Solutions

## Solution 1 — Compute NLL from probabilities

The correct class probability is `0.7`, so:

```text
NLL = -log(0.7) ~= 0.357
```

using the natural logarithm.

## Solution 2 — Derive cross-entropy from logits

```text
L = -log p_c
  = -log(exp(z_c) / sum_j exp(z_j))
  = -z_c + log(sum_j exp(z_j))
```

This is the logit form of multiclass cross-entropy.

## Solution 3 — Derive the logit gradient

Using `L = -z_c + log(sum_j exp(z_j))`:

```text
dL/dz_i = -1[i = c] + exp(z_i) / sum_j exp(z_j)
        = p_i - y_i
```

where `y_i` is one for the correct class and zero otherwise.

## Solution 4 — Compare reduction modes

The sum is `0.2 + 0.4 + 1.4 + 2.0 = 4.0`. The mean is `4.0 / 4 = 1.0`. Switching from `mean` to `sum` multiplies gradients by the batch size for this example, assuming the same per-example losses.

## Solution 5 — Choose the right loss

1. Next-token prediction: cross-entropy from logits because exactly one vocabulary token is correct.
2. Multi-label image tagging: binary cross-entropy because labels are independent and can co-occur.
3. Log-probability output for one of five classes: NLL loss because the model already returns log probabilities.
4. Binary fraud detector with one sigmoid output: binary cross-entropy because the output represents one independent Bernoulli label.
