# Gaussian Distribution

## Overview

The Gaussian distribution is the default continuous model for noise, measurement error, residuals, embeddings, and approximate uncertainty. It appears so often because sums and averages of many weakly dependent effects tend toward Gaussian behavior, and because its math is tractable.

In AI engineering, Gaussian assumptions connect least-squares regression, confidence intervals, initialization heuristics, variational methods, and anomaly scoring. This topic builds on [[probability-basics]] and feeds directly into [[mle-and-nll]], [[bayesian-statistics]], and [[bias-variance]].

## Math / Derivation

A univariate Gaussian random variable is written:

```text
X ~ N(mu, sigma^2)
```

with density:

```text
p(x) = 1 / sqrt(2 pi sigma^2) * exp(-(x - mu)^2 / (2 sigma^2))
```

Standardization converts it to a standard normal:

```text
Z = (X - mu) / sigma ~ N(0, 1)
```

For independent samples `x_1, ..., x_n` from `N(mu, sigma^2)`, the log-likelihood is:

```text
log p(x | mu, sigma^2)
  = -n/2 log(2 pi sigma^2) - (1 / (2 sigma^2)) sum_i (x_i - mu)^2
```

Maximizing this likelihood over `mu` gives the sample mean. With fixed variance, maximizing Gaussian likelihood is equivalent to minimizing squared error.

For vectors:

```text
X ~ N(mu, Sigma)
p(x) = 1 / sqrt((2 pi)^d |Sigma|)
       * exp(-1/2 (x - mu)^T Sigma^{-1} (x - mu))
```

The exponent uses Mahalanobis distance, which accounts for covariance scale and correlation.

## Intuition

A Gaussian is a symmetric bell-shaped distribution centered at `mu`, with spread controlled by `sigma`. Values near the mean are common; values many standard deviations away are exponentially unlikely.

The squared term in the exponent is why Gaussian noise leads to squared-error objectives. Large residuals are penalized quadratically, so Gaussian models are sensitive to outliers relative to heavy-tailed alternatives.

In multiple dimensions, covariance defines the shape of the uncertainty cloud. Diagonal covariance assumes independent axes; full covariance captures tilted ellipses where features move together.

## When & Why

Gaussian assumptions are useful when noise is roughly symmetric, aggregated from many small effects, and not too heavy-tailed. They are often a reasonable first approximation for residual diagnostics, embedding perturbations, and measurement noise.

Be careful when data is bounded, skewed, count-valued, multimodal, or outlier-heavy. In those cases, Gaussian likelihood can over-penalize rare but valid observations and produce brittle anomaly thresholds.

## Implementation

A later implementation pass should build Gaussian density and negative-log-likelihood calculations from scratch in PyTorch. It should compare manual formulas with distribution utilities, estimate `mu` and `sigma^2` from samples, and visualize how NLL changes with residual size.

The implementation should validate standardization, sample mean and variance estimates, Gaussian NLL, and Mahalanobis distance for a small covariance matrix.

## Cross-links

- `[[probability-basics]]` — defines densities, expectation, and variance.
- `[[mle-and-nll]]` — shows why Gaussian likelihood yields squared-error loss.
- `[[bayesian-statistics]]` — uses Gaussian priors and conjugate updates in common models.
- `[[linear-regression]]` — often assumes Gaussian residual noise.
- `[[bias-variance]]` — uses repeated-sampling behavior of estimators.

## Resources

- Bishop, "Pattern Recognition and Machine Learning." Springer, 2006.
- Murphy, "Probabilistic Machine Learning: An Introduction." MIT Press, 2022. <https://probml.github.io/pml-book/book1.html>
- Wasserman, "All of Statistics." Springer, 2004.
