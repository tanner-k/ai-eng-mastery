# Vanishing and Exploding Gradients — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Analyze a scalar chain

Let `h_t = w h_(t-1)` for `t = 1...T`. Derive `dh_T/dh_0`. For `T = 50`, compare the magnitude when `w = 0.9`, `w = 1.0`, and `w = 1.1`.

## Exercise 2 — Bound a Jacobian product

Suppose a 20-layer network has local Jacobian operator norms all bounded by `0.8`. Give an upper bound on the gradient norm at layer 1 relative to the output gradient norm.

## Exercise 3 — Diagnose training telemetry

A training run shows loss becoming `nan` after 200 steps. Gradient norm rises from `5` to `10,000` shortly before failure. What is the likely problem? Give three immediate mitigation steps.

## Exercise 4 — Explain saturation

Why can tanh saturation cause vanishing gradients? Use the derivative of tanh in your explanation.

## Exercise 5 — Compare mitigation strategies

For each intervention, state whether it primarily helps vanishing gradients, exploding gradients, or both: residual connections, gradient clipping, He initialization, and batch normalization.
