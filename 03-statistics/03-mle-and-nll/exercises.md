# MLE and Negative Log-Likelihood — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Bernoulli MLE

You observe binary labels `[1, 0, 1, 1, 0]` sampled from `Bernoulli(p)`. Derive the MLE for `p` and compute it for this data.

---

## Exercise 2 — Bernoulli NLL value

For labels `[1, 0, 1]` and predicted probabilities `[0.8, 0.3, 0.6]`, compute the Bernoulli NLL.

---

## Exercise 3 — Gaussian NLL and MSE

Assume targets `y = [2, 0]`, predictions `yhat = [1.5, 1]`, and fixed variance `sigma^2 = 1`. Compute the residual sum of squares and the Gaussian NLL up to the additive constant.

---

## Exercise 4 — Explain log-likelihood

Why do ML systems usually optimize log-likelihood rather than raw likelihood? Give one numerical reason and one optimization reason.

---

## Exercise 5 — Diagnose likelihood misspecification

A regression model trained with Gaussian NLL performs poorly on a dataset with many extreme outliers. Explain why the likelihood assumption may be the problem and suggest one alternative objective or likelihood.
