# Broadcasting in Neural Networks — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Determine broadcasted shapes

For each pair of shapes, state whether the operation is valid under trailing-dimension broadcasting. If valid, give the output shape.

1. `(32, 128) + (128,)`
2. `(16, 3, 224, 224) * (3, 1, 1)`
3. `(8, 10, 64) + (8, 64)`
4. `(4, 1, 7, 7) + (4, 3, 1, 1)`
5. `(5, 1) - (5,)`

## Exercise 2 — Derive the bias gradient

Let `Z = XW + b`, where `X` has shape `(B, Din)`, `W` has shape `(Din, Dout)`, and `b` has shape `(Dout,)`. Given an upstream gradient `G = dL/dZ` with shape `(B, Dout)`, derive `dL/db` and explain why the batch dimension is summed.

## Exercise 3 — Diagnose a silent loss bug

A model produces predictions with shape `(B, 1)`. Targets are loaded with shape `(B,)`. A learner computes `(pred - target) ** 2` and averages the result.

What shape does `pred - target` have? Why is this usually wrong? Give two robust fixes.

## Exercise 4 — Per-channel scaling

An image batch has shape `(B, C, H, W)`. You want to multiply each channel by a learned scale vector `gamma` with shape `(C,)`.

What shape should `gamma` be reshaped to before multiplication? Which axes will be broadcast? What shape should `dL/dgamma` have after backpropagation?

## Exercise 5 — Broadcasting versus repeating

Explain the difference between expanding a tensor with broadcasting and physically repeating its values. Include one case where broadcasting is preferable and one case where materializing repeated data may be necessary.
