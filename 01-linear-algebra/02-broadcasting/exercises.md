# Broadcasting — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Infer broadcasted shapes

For each pair of shapes, determine whether they are broadcast-compatible. If compatible, give the result shape. If not, identify the first incompatible aligned dimension.

1. `(32, 128)` and `(128,)`
2. `(10, 1, 5)` and `(1, 7, 5)`
3. `(4, 3)` and `(4,)`
4. `(2, 1, 8, 1)` and `(3, 1, 5)`
5. `(6, 1)` and `(1, 7)`

---

## Exercise 2 — Write the indexed formula

Let A have shape `(4, 1, 3)` and B have shape `(1, 5, 3)`. Define C = A + B.

1. What is the shape of C?
2. Write C[i, j, k] in terms of entries of A and B.
3. Which axes of A and B are broadcasted?

---

## Exercise 3 — Derive gradients through a broadcasted bias

Let Y = X + b, where X has shape `(n, d)` and b has shape `(d,)`. Let G = dL/dY.

Derive dL/dX and dL/db. Explain why dL/db requires a sum and name the axis being summed.

---

## Exercise 4 — Diagnose silent semantic broadcasting

A classifier produces logits with shape `(batch=16, classes=10)`. A learner wants to apply per-example weights and creates `weights` with shape `(16,)`, then computes:

```
weighted = logits * weights
```

1. Will this broadcast successfully?
2. If not, why?
3. What shape should `weights` have to apply one weight per example?
4. What shape would apply one weight per class instead?

---

## Exercise 5 — Compare expand and repeat

In PyTorch, a tensor b has shape `(3,)`. A learner writes both:

```
b_expanded = b.view(1, 3).expand(1000, 3)
b_repeated = b.view(1, 3).repeat(1000, 1)
```

Explain the difference in memory behavior. Which one is preferable for adding a bias to a `(1000, 3)` tensor, and why?
