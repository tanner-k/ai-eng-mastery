# Derivatives and Partials — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Derive a scalar derivative from the limit

Use the limit definition to derive the derivative of

```text
f(x) = 3x^2 - 2x + 5
```

Show the algebra from `(f(x + h) - f(x)) / h` to the final derivative.

---

## Exercise 2 — Compute partial derivatives of a model loss

For one training example, let

```text
L(w1, w2, b) = (w1 x1 + w2 x2 + b - y)^2
```

Derive `partial L / partial w1`, `partial L / partial w2`, and `partial L / partial b`. Then evaluate them at `x1 = 2`, `x2 = -1`, `y = 3`, `w1 = 0.5`, `w2 = -2`, and `b = 1`.

---

## Exercise 3 — Diagnose a finite-difference check

You are checking `f(x) = sin(x)` at `x = 1` using the centered finite difference

```text
(f(x + h) - f(x - h)) / (2h)
```

Explain why `h = 1e-1` and `h = 1e-12` can both be poor choices, even though one is much larger than the other. What range of `h` would you try first in float64?

---

## Exercise 4 — Identify nondifferentiable points

For each function, state where it is not differentiable and why:

1. `f(x) = abs(x)`
2. `g(x) = max(0, x)`
3. `h(x) = x^2 + abs(x - 1)`

Explain how this matters for models that use ReLU activations.

---

## Exercise 5 — Interpret partial derivatives in context

A validation metric `M(dropout, lr)` has local partial derivatives at the current configuration:

```text
partial M / partial dropout = -0.8
partial M / partial lr      = 12.0
```

Assume larger `M` is better and both hyperparameters are on their raw scales. What does each sign suggest locally? Why is it risky to compare the magnitudes directly?
