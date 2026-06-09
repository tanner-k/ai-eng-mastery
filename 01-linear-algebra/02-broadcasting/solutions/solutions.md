# Broadcasting — Solutions

---

## Solution 1 — Infer broadcasted shapes

1. `(32, 128)` and `(128,)` are compatible. Treat `(128,)` as `(1, 128)`, so the result is `(32, 128)`.
2. `(10, 1, 5)` and `(1, 7, 5)` are compatible. The result is `(10, 7, 5)`.
3. `(4, 3)` and `(4,)` are not compatible. Aligning from the right compares `3` with `4`, and neither is 1.
4. `(2, 1, 8, 1)` and `(3, 1, 5)` are compatible. Treat the second shape as `(1, 3, 1, 5)`, so the result is `(2, 3, 8, 5)`.
5. `(6, 1)` and `(1, 7)` are compatible. The result is `(6, 7)`.

---

## Solution 2 — Write the indexed formula

C has shape `(4, 5, 3)`.

The indexed formula is:

```
C[i, j, k] = A[i, 0, k] + B[0, j, k]
```

A is broadcast along axis 1, expanding from size 1 to size 5. B is broadcast along axis 0, expanding from size 1 to size 4.

---

## Solution 3 — Derive gradients through a broadcasted bias

Each output element is:

```
Y[i, j] = X[i, j] + b[j]
```

So:

```
dL/dX = G
```

because each X element influences exactly one Y element.

For b:

```
dL/db[j] = sum_{i=1}^n G[i, j]
```

or in vector form:

```
dL/db = G.sum(axis=0)
```

The sum is over the batch axis because each bias value b[j] is reused for every row in the batch.

---

## Solution 4 — Diagnose silent semantic broadcasting

`logits * weights` with shapes `(16, 10)` and `(16,)` does not broadcast successfully. Broadcasting aligns from the right, so it compares class dimension `10` with weight dimension `16`; neither is 1, so the operation fails.

For one weight per example, reshape weights to:

```
(16, 1)
```

Then `(16, 10) * (16, 1)` broadcasts across classes.

For one weight per class, use:

```
(10,)
```

or explicitly `(1, 10)`.

---

## Solution 5 — Compare expand and repeat

`expand` creates a view with stride 0 along the expanded dimension. It presents the tensor as shape `(1000, 3)` without copying the original three values 1000 times.

`repeat` physically copies the data into a new tensor with 3000 elements.

For adding a bias to a `(1000, 3)` tensor, `expand` or implicit broadcasting is preferable because no repeated storage is needed. Use `repeat` only when a real materialized copy is required for a later operation.
