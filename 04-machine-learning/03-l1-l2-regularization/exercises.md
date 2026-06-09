# L1 and L2 Regularization — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Derive the L2 gradient

For objective `J(w) = L(w) + lambda ||w||_2^2`, derive `grad J(w)` in terms of `grad L(w)`.

## Exercise 2 — Identify L1 subgradients

For `w = [-2, 0, 3]`, give one valid subgradient of `||w||_1`. Explain why the middle component is not unique.

## Exercise 3 — Compare penalty values

Compute L1 and squared L2 penalties for `w_a = [3, 0, 0]` and `w_b = [1, 1, 1]`. Which penalty prefers the spread-out vector?

## Exercise 4 — Choose a regularizer

A linear model has 50,000 sparse text features, and the product team wants a compact list of influential features. Would you choose L1, L2, or Elastic Net first? Explain.

## Exercise 5 — Diagnose over-regularization

Training loss and validation loss are both high, and increasing model size does not help while a large `lambda` is enabled. What failure mode is likely, and what would you try next?
