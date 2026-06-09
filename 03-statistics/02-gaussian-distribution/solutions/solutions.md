# Gaussian Distribution — Solutions

## Solution 1 — Standardize and interpret

`X ~ N(10, 4)` has `mu = 10` and `sigma = 2`.

```text
z = (13 - 10) / 2 = 1.5
```

The value is above the mean by `1.5` standard deviations.

## Solution 2 — Derive the MLE for the mean

The log-likelihood, ignoring constants independent of `mu`, is:

```text
ell(mu) = -(1 / (2 sigma^2)) sum_i (x_i - mu)^2
```

Differentiate:

```text
d ell / d mu = (1 / sigma^2) sum_i (x_i - mu)
```

Set to zero:

```text
sum_i (x_i - mu) = 0
sum_i x_i - n mu = 0
mu_hat = (1/n) sum_i x_i
```

The MLE is the sample mean.

## Solution 3 — Connect Gaussian NLL to MSE

For independent residuals `r_i = y_i - yhat_i` with fixed variance:

```text
NLL = n/2 log(2 pi sigma^2) + (1 / (2 sigma^2)) sum_i r_i^2
```

The first term is constant with respect to predictions, and `1 / (2 sigma^2)` is a positive constant. Therefore minimizing NLL over predictions is equivalent to minimizing:

```text
sum_i (y_i - yhat_i)^2
```

## Solution 4 — Compute a diagonal Gaussian log density

Coordinate contributions:

```text
j = 1: (1 - 0)^2 / 1 = 1
j = 2: (3 - 1)^2 / 4 = 1
```

The squared standardized residual sum is:

```text
1 + 1 = 2
```

Both coordinates contribute equally to the exponent after accounting for variance.

## Solution 5 — Diagnose a bad Gaussian assumption

Gaussian latency modeling can assign unrealistically tiny probabilities to legitimate tail events, causing excessive anomaly alerts. It can also fit inflated variance to accommodate spikes, making the detector insensitive to moderate but important regressions.

Alternatives include modeling log-latency as Gaussian, using a log-normal distribution, using quantile-based thresholds, or using a heavy-tailed distribution.
