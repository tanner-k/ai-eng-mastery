# Matrix Multiplication — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Compute a product by hand

Let

```
A = [[1, 2, 0],
     [-1, 3, 4]]

B = [[2, 1],
     [0, -2],
     [5, 3]]
```

Compute C = AB. State the shape of C and show the dot product that produces C_22.

---

## Exercise 2 — Reason about valid shapes

For each expression, say whether it is valid. If valid, give the output shape. If invalid, explain which dimensions fail to match.

1. `A @ B`, where A has shape `(32, 128)` and B has shape `(128, 64)`.
2. `B @ A`, using the same A and B.
3. `Q @ K.transpose(-2, -1)`, where Q and K each have shape `(8, 12, 256, 64)`.
4. `X @ W + b`, where X has shape `(16, 10)`, W has shape `(11, 4)`, and b has shape `(4,)`.

---

## Exercise 3 — Derive linear-layer gradients

Let Y = XW + b, where X is in R^(n x d), W is in R^(d x k), b is in R^k, and Y is in R^(n x k). Given upstream gradient G = dL/dY, derive dL/dX, dL/dW, and dL/db. Include shapes for each result.

---

## Exercise 4 — Choose an efficient multiplication order

Suppose A has shape `(1000, 10)`, B has shape `(10, 1000)`, and C has shape `(1000, 5)`.

Both `(A @ B) @ C` and `A @ (B @ C)` are valid. Estimate the number of scalar multiply-adds for each parenthesization and choose the cheaper order.

---

## Exercise 5 — Diagnose a model-shape bug

A model receives a batch X with shape `(batch=64, features=768)` and should produce logits for 20 classes. A learner creates W with shape `(20, 768)` and writes `logits = X @ W`.

1. Why does this fail?
2. What are two correct fixes?
3. Which fix best matches the usual PyTorch `nn.Linear` parameter convention, and why?
