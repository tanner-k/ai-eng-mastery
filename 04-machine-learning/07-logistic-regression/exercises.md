# Logistic Regression — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Convert logit to probability

Compute `sigmoid(z)` for logits `z = -2, 0, 2`. Interpret the results.

## Exercise 2 — Derive the BCE gradient

For one example with logit `z`, probability `p = sigmoid(z)`, and label `y`, show that the gradient of binary cross-entropy with respect to `z` is `p - y`.

## Exercise 3 — Interpret coefficients

A logistic regression model has coefficient `w_j = 0.7` for a standardized feature. What happens to the odds when that feature increases by one standard deviation?

## Exercise 4 — Handle separable data

Why can logistic regression coefficients grow without bound on perfectly linearly separable data? What prevents this in practice?

## Exercise 5 — Tune a threshold

A model predicts probabilities for rare positive cases. At threshold `0.5`, recall is too low. What threshold change would you try, and what metric tradeoff do you expect?
