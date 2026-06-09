# Backpropagation — Solutions

## Solution 1 — Derive gradients for a one-hidden-layer network

Let `e = y_hat - y`. Then:

```text
dL/dy_hat = e
dL/dW2 = h^T e
dL/db2 = e
dh = e W2^T
dz1 = dh * (1 - tanh(z1)^2)
dL/dW1 = x^T dz1
dL/db1 = dz1
```

where `z1 = xW1 + b1` and `h = tanh(z1)`.

## Solution 2 — Add the batch dimension

With `E = (Yhat - Y)` for a summed squared loss:

```text
dW2: (Hdim, Dout) = H^T E
db2: (Dout,) = E.sum(axis=0)
dH:  (B, Hdim) = E W2^T
dZ1: (B, Hdim) = dH * activation'(Z1)
dW1: (Din, Hdim) = X^T dZ1
db1: (Hdim,) = dZ1.sum(axis=0)
```

If the loss is averaged over batch, include the corresponding `1/B` factor.

## Solution 3 — Count work for finite differences

Central differences require `2 * 10,000,000 = 20,000,000` forward passes. At 20 ms each, that is 400,000 seconds, or about 111 hours. Backpropagation computes all gradients in roughly one forward plus one backward pass, typically on the order of milliseconds to seconds for the same model depending on hardware.

## Solution 4 — Identify cached values

Cache `X` and `W` for the affine backward pass, `Z` or the ReLU mask `Z > 0` for the activation backward pass, and the reduction scale for `mean(A)`. `b` is not usually needed to compute its own gradient, but its shape is needed to reduce correctly.

## Solution 5 — Diagnose a broken gradient

Plausible causes:

- A saturated activation such as tanh with very large preactivations. Check activation histograms and local derivatives.
- Dead ReLU units in the first layer. Check the fraction of `Z <= 0` and try smaller initialization or leaky ReLU.
- A graph break such as `.detach()`, conversion to NumPy, or an in-place overwrite. Check whether first-layer parameters have non-`None` gradients and inspect the computation path.
