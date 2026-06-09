# Bayesian Statistics — Solutions

## Solution 1 — Beta-Bernoulli update

For a Beta prior and Bernoulli data:

```text
Beta(alpha, beta) + s successes + f failures
  -> Beta(alpha + s, beta + f)
```

With `Beta(2, 2)`, `s = 7`, and `f = 3`:

```text
posterior = Beta(9, 5)
```

The posterior mean is:

```text
9 / (9 + 5) = 9/14 approx 0.6429
```

## Solution 2 — Compare MLE and posterior mean

The Bernoulli MLE is the observed success rate:

```text
p_hat_MLE = 7 / 10 = 0.7
```

The posterior mean is `9/14 approx 0.6429`. It is lower because the `Beta(2, 2)` prior contributes two prior successes and two prior failures, pulling the estimate toward `0.5`.

## Solution 3 — MAP as regularized likelihood

For `w ~ N(0, tau^2)`:

```text
p(w) = 1 / sqrt(2 pi tau^2) * exp(-w^2 / (2 tau^2))
```

The negative log prior is:

```text
-log p(w) = constant + w^2 / (2 tau^2)
```

The negative log posterior is:

```text
-log p(D | w) - log p(w)
```

so the Gaussian prior adds an L2 penalty proportional to `w^2`.

## Solution 4 — Interpret posterior uncertainty

Both means are:

```text
Beta(60, 40): 60 / 100 = 0.60
Beta(6, 4):   6 / 10   = 0.60
```

Model B has more uncertainty because it has much lower concentration: `alpha + beta = 10` rather than `100`. The mean is the same, but Model A represents much more evidence around that mean, so its posterior is narrower.

## Solution 5 — Use posterior predictive reasoning

From Exercise 1, the posterior is `Beta(9, 5)`. The posterior predictive probability of success is:

```text
E[p | D] = 9 / (9 + 5) = 9/14 approx 0.6429
```

So the next success probability is about `64.3%`.
