# Gradients and Jacobians — Solutions

## Solution 1 — Compute a gradient

For

```text
f(x, y, z) = x^2 y + sin(z) - 3xy
```

the partial derivatives are:

```text
partial f / partial x = 2xy - 3y
partial f / partial y = x^2 - 3x
partial f / partial z = cos(z)
```

At `(2, -1, pi/2)`:

```text
partial f / partial x = 2(2)(-1) - 3(-1) = -1
partial f / partial y = 2^2 - 3(2) = -2
partial f / partial z = cos(pi/2) = 0
```

So:

```text
grad f = [-1, -2, 0]^T
```

## Solution 2 — Compute a Jacobian and its shape

The outputs are:

```text
F1 = x1 + x2
F2 = x1 x2
F3 = exp(x1 - x2)
```

The Jacobian has one row per output and one column per input:

```text
J = [[1,                    1],
     [x2,                   x1],
     [exp(x1 - x2), -exp(x1 - x2)]]
```

Its shape is `3 x 2`.

## Solution 3 — Use a Jacobian for a local approximation

At `(0, 0)`,

```text
J = [[1,  1],
     [0,  0],
     [1, -1]]
```

For `dx = [0.1, -0.2]^T`:

```text
J dx = [-0.1, 0, 0.3]^T
```

The original output is `F(0, 0) = [0, 0, 1]`, so the linear approximation predicts:

```text
[-0.1, 0, 1.3]
```

The exact output at `(0.1, -0.2)` is:

```text
[ -0.1, -0.02, exp(0.3) ] approx [-0.1, -0.02, 1.3499]
```

The approximation is directionally right but misses second-order effects, especially in the product and exponential terms.

## Solution 4 — Relate a VJP to backpropagation

Compute:

```text
dL/dx = (dL/dy) J
```

With `dL/dy = [0.1, -0.2, 0.5]`:

```text
first component  = 0.1(1) + (-0.2)(3) + 0.5(5) = 2.0
second component = 0.1(2) + (-0.2)(4) + 0.5(6) = 2.4
```

So:

```text
dL/dx = [2.0, 2.4]
```

## Solution 5 — Explain why full Jacobians are expensive

Per example, the Jacobian from a `4096`-dimensional representation to `50,000` logits has:

```text
50,000 * 4,096 = 204,800,000 entries
```

For a batch of `32`, a naive per-example collection would contain:

```text
32 * 204,800,000 = 6,553,600,000 entries
```

That is billions of numbers for one layer interface. During training, the scalar loss supplies an upstream gradient, so a vector-Jacobian product can compute the needed representation gradient directly without storing the full matrix.
