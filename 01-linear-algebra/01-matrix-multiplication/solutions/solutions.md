# Matrix Multiplication — Solutions

---

## Solution 1 — Compute a product by hand

A has shape `(2, 3)` and B has shape `(3, 2)`, so C = AB has shape `(2, 2)`.

```
C_11 = 1*2 + 2*0 + 0*5 = 2
C_12 = 1*1 + 2*(-2) + 0*3 = -3
C_21 = (-1)*2 + 3*0 + 4*5 = 18
C_22 = (-1)*1 + 3*(-2) + 4*3 = 5
```

So:

```
C = [[2, -3],
     [18, 5]]
```

The requested entry C_22 is the dot product of row 2 of A with column 2 of B:

```
[-1, 3, 4] dot [1, -2, 3] = 5
```

---

## Solution 2 — Reason about valid shapes

1. `A @ B` is valid because the inner dimensions match: `(32, 128) @ (128, 64) -> (32, 64)`.
2. `B @ A` is invalid because `(128, 64) @ (32, 128)` has inner dimensions `64` and `32`, which do not match.
3. Valid. `K.transpose(-2, -1)` has shape `(8, 12, 64, 256)`, so the batched product has shape `(8, 12, 256, 256)`.
4. Invalid before the bias matters. `(16, 10) @ (11, 4)` fails because `10 != 11`.

---

## Solution 3 — Derive linear-layer gradients

For Y = XW + b and upstream gradient G = dL/dY:

```
dL/dX = G W^T
```

Shape:

```
(n x k) @ (k x d) -> (n x d)
```

For W, each W_ij contributes to every example through X rows:

```
dL/dW = X^T G
```

Shape:

```
(d x n) @ (n x k) -> (d x k)
```

For b, the same bias vector is added to every row, so gradients accumulate over the batch:

```
dL/db = sum_{r=1}^n G_r
```

Shape: `(k,)`.

---

## Solution 4 — Choose an efficient multiplication order

For `(A @ B) @ C`:

```
A @ B: (1000 x 10) @ (10 x 1000) -> (1000 x 1000)
cost = 1000 * 10 * 1000 = 10,000,000

(A @ B) @ C: (1000 x 1000) @ (1000 x 5) -> (1000 x 5)
cost = 1000 * 1000 * 5 = 5,000,000

total = 15,000,000
```

For `A @ (B @ C)`:

```
B @ C: (10 x 1000) @ (1000 x 5) -> (10 x 5)
cost = 10 * 1000 * 5 = 50,000

A @ (B @ C): (1000 x 10) @ (10 x 5) -> (1000 x 5)
cost = 1000 * 10 * 5 = 50,000

total = 100,000
```

`A @ (B @ C)` is cheaper by a factor of 150 and avoids materializing a `(1000, 1000)` intermediate.

---

## Solution 5 — Diagnose a model-shape bug

`X @ W` fails because X has shape `(64, 768)` and W has shape `(20, 768)`. The inner dimensions are `768` and `20`, not equal.

Two correct fixes are:

```
W = shape (768, 20)
logits = X @ W
```

or:

```
W = shape (20, 768)
logits = X @ W.T
```

The second fix matches the usual PyTorch `nn.Linear` convention. PyTorch stores linear-layer weights as `(out_features, in_features)`, so its forward pass effectively computes `X @ weight.T + bias`.
