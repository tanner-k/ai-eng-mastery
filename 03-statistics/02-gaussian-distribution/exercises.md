# Gaussian Distribution — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Standardize and interpret

If `X ~ N(10, 4)`, compute the z-score for `x = 13`. Is this value above or below the mean, and by how many standard deviations?

---

## Exercise 2 — Derive the MLE for the mean

Assume `x_1, ..., x_n` are independent samples from `N(mu, sigma^2)` and `sigma^2` is known. Derive the maximum-likelihood estimate of `mu`.

---

## Exercise 3 — Connect Gaussian NLL to MSE

For fixed `sigma^2`, show why minimizing Gaussian negative log-likelihood over predictions `yhat_i` is equivalent to minimizing sum of squared errors.

---

## Exercise 4 — Compute a diagonal Gaussian log density

Let `x = [1, 3]`, `mu = [0, 1]`, and diagonal variances `[1, 4]`. Compute the squared standardized residual sum:

```text
sum_j (x_j - mu_j)^2 / sigma_j^2
```

Then state which coordinate contributes more to the exponent.

---

## Exercise 5 — Diagnose a bad Gaussian assumption

A latency model assumes request latency is Gaussian, but observed latencies are positive, right-skewed, and have occasional large spikes. What problems can this cause for anomaly detection? Name one alternative modeling choice.
