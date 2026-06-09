# Gradients and Jacobians — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Compute a gradient

Let

```text
f(x, y, z) = x^2 y + sin(z) - 3xy
```

Compute `grad f`. Then evaluate it at `(x, y, z) = (2, -1, pi/2)`.

---

## Exercise 2 — Compute a Jacobian and its shape

Let `F: R^2 -> R^3` be

```text
F(x1, x2) = [x1 + x2, x1 x2, exp(x1 - x2)]
```

Compute the Jacobian and state its shape.

---

## Exercise 3 — Use a Jacobian for a local approximation

For the function in Exercise 2, estimate the change in `F` at `(0, 0)` for a small perturbation `dx = [0.1, -0.2]^T` using the Jacobian. Then compute the exact new output and compare qualitatively.

---

## Exercise 4 — Relate a VJP to backpropagation

Suppose `y = F(x)` with Jacobian

```text
J = [[1, 2],
     [3, 4],
     [5, 6]]
```

where `F: R^2 -> R^3`. If the upstream gradient for scalar loss `L` is `dL/dy = [0.1, -0.2, 0.5]`, compute `dL/dx`.

---

## Exercise 5 — Explain why full Jacobians are expensive

A model maps a batch of `32` examples to `50,000` logits, and each example representation has dimension `4096`. How many entries would a per-example full Jacobian from representation to logits contain? Why is a vector-Jacobian product usually preferable during training?
