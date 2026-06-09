# MLE and Negative Log-Likelihood — Solutions

## Solution 1 — Bernoulli MLE

For Bernoulli data, the likelihood is:

```text
L(p) = product_i p^{y_i}(1 - p)^{1 - y_i}
```

The MLE is the sample fraction of ones:

```text
p_hat = (1/n) sum_i y_i
```

For `[1, 0, 1, 1, 0]`, there are `3` ones out of `5`, so:

```text
p_hat = 3/5 = 0.6
```

## Solution 2 — Bernoulli NLL value

Bernoulli NLL is:

```text
-sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)]
```

For the examples:

```text
NLL = -[log(0.8) + log(1 - 0.3) + log(0.6)]
    = -[log(0.8) + log(0.7) + log(0.6)]
    approx 1.0906
```

## Solution 3 — Gaussian NLL and MSE

Residuals:

```text
2 - 1.5 = 0.5
0 - 1 = -1
```

Residual sum of squares:

```text
0.5^2 + (-1)^2 = 0.25 + 1 = 1.25
```

With `sigma^2 = 1`, Gaussian NLL up to the additive constant is:

```text
(1 / 2) * 1.25 = 0.625
```

## Solution 4 — Explain log-likelihood

Numerically, raw likelihood multiplies many probabilities less than one, which can underflow to zero. Log-likelihood sums log probabilities, which is much more stable.

For optimization, log-likelihood turns products into sums of per-example terms. This makes gradients additive across minibatches and easier to compute.

## Solution 5 — Diagnose likelihood misspecification

Gaussian NLL penalizes residuals quadratically, so extreme outliers can dominate training. If the data is heavy-tailed, the model may inflate variance, chase outliers, or fit the typical cases poorly.

Alternatives include a Laplace likelihood, Huber loss, quantile loss, or a Student-t likelihood.
