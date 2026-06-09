# Batch Normalization — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Normalize a small batch

For one feature with batch values `[1, 3, 5]`, compute the batch mean, variance using denominator `m`, normalized values with `eps = 0`, and outputs when `gamma = 2` and `beta = -1`.

## Exercise 2 — State parameter shapes

Give the shapes of `gamma` and `beta` for:

1. Dense activations with shape `(B, D)`.
2. Image activations with shape `(B, C, H, W)`.

Also state which axes are reduced to compute mean and variance.

## Exercise 3 — Training versus inference

Explain why batch norm should use batch statistics during training but running statistics during inference. What can go wrong if a model is evaluated in training mode?

## Exercise 4 — Small-batch failure

A model with batch norm trains with batch size 2 and validation accuracy is highly unstable. Explain why batch norm may be contributing and name two alternatives or mitigations.

## Exercise 5 — Bias before batch norm

Consider a linear layer `z = xW + b` followed immediately by batch norm with learnable `beta`. Explain why the bias `b` is often redundant.
