# Bias-Variance Tradeoff — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Identify bias and variance regimes

For each situation, state whether high bias, high variance, or both are likely:

1. Training error is high and validation error is high.
2. Training error is near zero and validation error is much higher.
3. Both training and validation error are low.
4. Training error is moderate, validation error is high, and results vary greatly across random seeds.

---

## Exercise 2 — Compute a simple decomposition

At a fixed input `x`, the true function value is `f(x) = 10`. Across repeated training sets, a model's predictions have mean `8` and variance `3`. Observation noise variance is `2`. Compute squared bias and expected squared prediction error.

---

## Exercise 3 — Reason about model capacity

You fit polynomial regression models of degree `1`, `3`, and `20` to noisy samples from a smooth cubic function. Which degree is most likely high bias? Which is most likely high variance? Which is likely best, assuming enough but not unlimited data?

---

## Exercise 4 — Explain ensembling

Why can averaging predictions from many independently trained high-variance models reduce variance? What assumption makes the reduction strongest?

---

## Exercise 5 — Diagnose a production regression

A model's offline validation metric is strong, but production performance drops sharply on a new traffic segment. Is this necessarily a bias-variance problem? Explain at least two possible causes and one diagnostic.
