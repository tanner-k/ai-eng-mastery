# Broadcasting in Neural Networks — Solutions

## Solution 1 — Determine broadcasted shapes

1. Valid: `(32, 128)`.
2. Valid: `(16, 3, 224, 224)`.
3. Invalid: align from the right, so `10` and `8` conflict.
4. Valid: `(4, 3, 7, 7)`.
5. Valid but usually suspicious: `(5, 5)`, because `(5,)` aligns as `(1, 5)`.

## Solution 2 — Derive the bias gradient

For `Z[b, j] = sum_i X[b, i] W[i, j] + b[j]`, each `b[j]` contributes to every row `Z[b, j]`. Therefore:

```text
dL/db[j] = sum over b of dL/dZ[b, j]
```

In vector form, `dL/db = G.sum(axis=0)`, with shape `(Dout,)`.

## Solution 3 — Diagnose a silent loss bug

`pred` has shape `(B, 1)` and `target` has shape `(B,)`, which is treated as `(1, B)`. The subtraction broadcasts to `(B, B)`. This compares every prediction with every target instead of matching examples one-to-one.

Two robust fixes are to reshape targets to `(B, 1)` or squeeze predictions to `(B,)` before subtraction. Also add an assertion such as `assert pred.shape == target.shape` before computing elementwise loss.

## Solution 4 — Per-channel scaling

Reshape `gamma` to `(1, C, 1, 1)`. It broadcasts across batch, height, and width while matching channel. The upstream gradient with shape `(B, C, H, W)` must be summed over axes `0`, `2`, and `3`, giving `dL/dgamma` with shape `(C,)`.

## Solution 5 — Broadcasting versus repeating

Broadcasting reuses values virtually through shape and stride metadata. Repeating physically materializes copies. Broadcasting is preferable for adding a bias vector to a large batch because it avoids extra memory. Materializing may be necessary when an operation requires contiguous storage or when you need independent mutable copies of repeated values.
