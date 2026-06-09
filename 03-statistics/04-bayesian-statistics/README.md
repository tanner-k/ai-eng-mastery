# Bayesian Statistics

## Overview

Bayesian statistics updates beliefs with evidence. Instead of treating parameters as fixed unknown constants and returning only a point estimate, Bayesian inference represents uncertainty about parameters with probability distributions.

In AI engineering, Bayesian thinking helps with uncertainty estimation, data scarcity, online updating, calibration, exploration, and regularization. Even when full Bayesian inference is too expensive, its concepts explain priors, posterior predictive distributions, MAP estimates, and why model confidence should change with evidence.

This topic builds on [[probability-basics]] and [[mle-and-nll]], and it connects to [[gaussian-distribution]] and [[bias-variance]].

## Math / Derivation

Bayes' rule for parameters `theta` and data `D` is:

```text
p(theta | D) = p(D | theta) p(theta) / p(D)
```

where:

- `p(theta)` is the prior.
- `p(D | theta)` is the likelihood.
- `p(theta | D)` is the posterior.
- `p(D)` is the evidence or marginal likelihood.

The posterior predictive distribution integrates over parameter uncertainty:

```text
p(y_new | x_new, D) = integral p(y_new | x_new, theta) p(theta | D) d theta
```

For a Beta-Bernoulli model:

```text
p ~ Beta(alpha, beta)
y_i ~ Bernoulli(p)
```

after observing `s` successes and `f` failures:

```text
p | D ~ Beta(alpha + s, beta + f)
```

The posterior mean is:

```text
E[p | D] = (alpha + s) / (alpha + beta + s + f)
```

Maximum a posteriori (MAP) estimation chooses the posterior mode:

```text
theta_MAP = argmax_theta [log p(D | theta) + log p(theta)]
```

## Intuition

Bayesian inference treats learning as belief revision. The prior says what was plausible before seeing data. The likelihood says how compatible each parameter value is with the observed data. The posterior combines both.

With little data, the prior matters. With lots of representative data, the likelihood usually dominates. This is why Bayesian methods are valuable in low-data or high-stakes settings where uncertainty should remain visible.

Regularization has a Bayesian interpretation: L2 penalties correspond to Gaussian priors on weights, and L1 penalties correspond to Laplace priors.

## When & Why

Use Bayesian reasoning when uncertainty matters as much as point prediction:

- Estimating click-through rates with small sample sizes.
- Ranking experiments while accounting for uncertainty.
- Thompson sampling and exploration.
- Calibrating risk in medical, safety, or abuse-detection systems.
- Understanding regularization as prior information.

The main cost is computation. Exact posterior inference is often impossible for large neural networks, so practical systems use approximations, ensembles, variational methods, or Bayesian thinking applied to simpler components.

## Implementation

A later implementation pass should implement a Beta-Bernoulli updater and compare MLE, MAP, posterior mean, and posterior predictive intervals over sequential observations. It should also demonstrate how a Gaussian prior adds an L2-style penalty to a simple likelihood objective.

The implementation should validate conjugate updates, posterior means, credible intervals by sampling or quantiles, and the effect of prior strength under small and large datasets.

## Cross-links

- `[[probability-basics]]` — provides conditional probability and Bayes' rule.
- `[[mle-and-nll]]` — supplies the likelihood part of Bayesian inference.
- `[[gaussian-distribution]]` — common prior, likelihood, and approximation family.
- `[[bias-variance]]` — priors can reduce variance by adding structure.
- `[[l1-l2-regularization]]` — regularization has prior interpretations.

## Resources

- Gelman et al., "Bayesian Data Analysis." CRC Press, 3rd edition.
- Murphy, "Probabilistic Machine Learning: An Introduction." MIT Press, 2022. <https://probml.github.io/pml-book/book1.html>
- David MacKay, "Information Theory, Inference, and Learning Algorithms." <http://www.inference.org.uk/itila/book.html>
