# Activations: tanh and ReLU — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Derive the tanh derivative

Starting from `tanh(x) = sinh(x) / cosh(x)`, derive `d tanh(x)/dx = 1 - tanh(x)^2`.

## Exercise 2 — Compute activation outputs and derivatives

For inputs `[-3, -1, 0, 1, 3]`, describe the ReLU outputs and derivatives. Then describe qualitatively where tanh derivatives are largest and smallest.

## Exercise 3 — Explain why linear layers need nonlinear activations

Show that two linear layers without an activation between them are equivalent to one linear layer.

## Exercise 4 — Diagnose dead ReLUs

A hidden layer has preactivations that are negative for 98% of examples throughout training. What symptoms would appear in gradients and activations? Name two remedies.

## Exercise 5 — Match activation and initialization

For a deep MLP using ReLU, explain why He initialization is more appropriate than a naive small random initialization. What problem is it trying to prevent?
