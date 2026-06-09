# Batch Normalization — Solutions

## Solution 1 — Normalize a small batch

Mean is `(1 + 3 + 5) / 3 = 3`. Variance with denominator `m` is `((1 - 3)^2 + (3 - 3)^2 + (5 - 3)^2) / 3 = 8/3`.

Normalized values are:

```text
[-2/sqrt(8/3), 0, 2/sqrt(8/3)] ~= [-1.225, 0, 1.225]
```

With `gamma = 2` and `beta = -1`, outputs are approximately `[-3.449, -1.000, 1.449]`.

## Solution 2 — State parameter shapes

For dense activations `(B, D)`, `gamma` and `beta` have shape `(D,)`; statistics reduce over batch axis `B`.

For image activations `(B, C, H, W)`, `gamma` and `beta` have shape `(C,)` or are reshaped to `(1, C, 1, 1)`; statistics reduce over batch, height, and width for each channel.

## Solution 3 — Training versus inference

Training uses batch statistics so normalization reflects the current mini-batch and remains differentiable through the batch. Inference uses running statistics for deterministic outputs and to avoid depending on the other examples in a request batch. Evaluating in training mode can make predictions batch-dependent and noisy, and it may update running statistics incorrectly.

## Solution 4 — Small-batch failure

With batch size 2, mean and variance estimates are very noisy. This noise changes normalized activations and running statistics, causing unstable validation behavior. Mitigations include larger batches, synchronized batch norm, group norm, layer norm, freezing running stats, or removing batch norm.

## Solution 5 — Bias before batch norm

Batch norm subtracts the batch mean of `z = xW + b`. The constant bias shifts every example and is removed by mean subtraction. The learnable `beta` after normalization can provide any needed shift, so the pre-batch-norm bias is often redundant.
