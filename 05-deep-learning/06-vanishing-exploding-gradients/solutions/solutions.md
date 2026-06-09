# Vanishing and Exploding Gradients — Solutions

## Solution 1 — Analyze a scalar chain

Since `h_T = w^T h_0`, the derivative is:

```text
dh_T/dh_0 = w^T
```

For `T = 50`: `0.9^50 ~= 0.005`, `1.0^50 = 1`, and `1.1^50 ~= 117.4`. These show vanishing, stable, and exploding behavior.

## Solution 2 — Bound a Jacobian product

The relative norm is bounded by `0.8^20 ~= 0.0115`. The layer-1 gradient norm is at most about 1.15% of the output gradient norm under that bound.

## Solution 3 — Diagnose training telemetry

The likely problem is exploding gradients. Immediate mitigations include lowering the learning rate, applying global norm clipping, improving initialization or normalization, and checking for unstable loss scaling in mixed precision.

## Solution 4 — Explain saturation

When `|x|` is large, `tanh(x)` is close to `-1` or `1`, so `1 - tanh(x)^2` is close to zero. Backprop multiplies by this derivative, so saturated units strongly dampen the upstream gradient.

## Solution 5 — Compare mitigation strategies

Residual connections primarily help vanishing gradients and can also stabilize deep training. Gradient clipping primarily helps exploding gradients. He initialization helps both by preserving variance in ReLU networks. Batch normalization can help both by stabilizing activation distributions and local gradient scale.
