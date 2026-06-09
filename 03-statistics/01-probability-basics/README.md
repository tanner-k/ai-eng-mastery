# Probability Basics

## Overview

Probability is the language of uncertainty in AI systems. Data is sampled from populations, labels are noisy, models output distributions, and evaluation metrics are estimates rather than fixed truths. A strong probability foundation lets you reason about what a model knows, what it assumes, and how much evidence supports a decision.

In engineering practice, probability shows up in train/test splits, calibration, A/B tests, generative models, Bayesian updates, and likelihood-based objectives. It is also the foundation for [[gaussian-distribution]], [[mle-and-nll]], [[bayesian-statistics]], and [[bias-variance]].

## Math / Derivation

A probability space assigns probabilities to events with:

```text
0 <= P(A) <= 1
P(S) = 1
P(A union B) = P(A) + P(B) for disjoint A, B
```

For events `A` and `B`:

```text
P(A | B) = P(A and B) / P(B), when P(B) > 0
P(A and B) = P(A | B) P(B)
```

Independence means:

```text
P(A and B) = P(A) P(B)
```

For a discrete random variable `X`, expectation and variance are:

```text
E[X] = sum_x x P(X = x)
Var(X) = E[(X - E[X])^2] = E[X^2] - E[X]^2
```

Bayes' rule follows from the product rule:

```text
P(A | B) = P(B | A) P(A) / P(B)
```

## Intuition

Probability is disciplined bookkeeping for uncertainty. Conditional probability updates the sample space after evidence arrives. Independence says that learning one event happened does not change the probability of the other event.

Expectation is the long-run average value under repeated sampling. Variance measures spread around that average. In machine learning, a model's single metric on one validation set is a random variable; it depends on which examples were sampled.

## When & Why

Use probability reasoning when:

- Estimating whether metric changes are signal or sampling noise.
- Interpreting classifier probabilities and calibration.
- Deriving likelihoods for [[mle-and-nll]].
- Understanding priors, posteriors, and uncertainty in [[bayesian-statistics]].
- Explaining why more validation data reduces metric variance.

The key failure mode is treating estimates as certainties. A 0.5% accuracy lift may be meaningful on millions of examples and meaningless on a tiny validation set.

## Implementation

A later implementation pass should build simulations that connect probability formulas to empirical frequencies. It should sample Bernoulli and categorical variables, estimate expectations and variances, and show how estimates converge as sample size grows.

The implementation should validate conditional probability, independence checks, empirical expectation, variance estimates, and confidence intervals for simple metrics.

## Cross-links

- `[[gaussian-distribution]]` — a central continuous distribution built from probability basics.
- `[[mle-and-nll]]` — turns probability models into training objectives.
- `[[bayesian-statistics]]` — uses Bayes' rule to update beliefs.
- `[[bias-variance]]` — studies how estimator error decomposes under sampling.

## Resources

- Blitzstein and Hwang, "Introduction to Probability." <https://projects.iq.harvard.edu/stat110/home>
- Wasserman, "All of Statistics." Springer, 2004.
- Murphy, "Probabilistic Machine Learning: An Introduction." MIT Press, 2022. <https://probml.github.io/pml-book/book1.html>
