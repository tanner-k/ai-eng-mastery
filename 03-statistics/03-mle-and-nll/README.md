# MLE and Negative Log-Likelihood

## Overview

Maximum likelihood estimation (MLE) chooses parameters that make the observed data most probable under a chosen model. Negative log-likelihood (NLL) turns that probability objective into a minimization problem suitable for optimization.

This is the bridge between probability and training loss. Gaussian likelihood leads to squared error, Bernoulli likelihood leads to binary cross-entropy, categorical likelihood leads to multiclass cross-entropy, and sequence models train by minimizing token-level NLL. This topic builds on [[probability-basics]] and [[gaussian-distribution]], then connects to [[loss-functions]], [[cross-entropy-nll]], and [[logistic-regression]].

## Math / Derivation

Given data `D = {x_1, ..., x_n}` and model parameters `theta`, the likelihood is:

```text
L(theta; D) = p(D | theta)
```

For independent observations:

```text
L(theta; D) = product_i p(x_i | theta)
```

MLE chooses:

```text
theta_hat = argmax_theta L(theta; D)
```

Because products can underflow and logs turn products into sums, we usually maximize log-likelihood:

```text
ell(theta; D) = log L(theta; D) = sum_i log p(x_i | theta)
```

Training frameworks usually minimize negative log-likelihood:

```text
NLL(theta; D) = -sum_i log p(x_i | theta)
```

Bernoulli example for labels `y_i in {0, 1}` and predicted probability `p_i`:

```text
p(y_i | p_i) = p_i^{y_i} (1 - p_i)^{1 - y_i}
NLL = -sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)]
```

This is binary cross-entropy.

## Intuition

Likelihood asks: "If my model were true, how unsurprising would this data be?" MLE picks the parameter setting that makes the observed data least surprising.

The negative log is not just a mathematical convenience. It converts multiplying many probabilities into adding per-example penalties. It also heavily penalizes confident wrong predictions because `-log(p)` grows rapidly as assigned probability approaches zero.

## When & Why

Use MLE/NLL when model outputs are probabilistic and you can write down the probability of observed targets. It gives a principled loss function and often produces well-behaved gradients.

Common examples:

- Regression with Gaussian noise gives squared-error loss.
- Binary classification with Bernoulli labels gives binary cross-entropy.
- Multiclass classification with categorical labels gives cross-entropy.
- Language modeling minimizes NLL of the next token.

The main risk is model misspecification. MLE is only as good as the assumed likelihood family; wrong noise assumptions can produce brittle estimates.

## Implementation

A later implementation pass should implement Bernoulli and Gaussian NLL from formulas, compare them with PyTorch loss functions, and fit simple parameters by gradient descent. It should include numerical-stability checks such as clamping probabilities or using logits directly.

The implementation should validate equivalence between Gaussian NLL and MSE under fixed variance, Bernoulli NLL and binary cross-entropy, and categorical NLL and cross-entropy over logits.

## Cross-links

- `[[probability-basics]]` — defines likelihoods and conditional probability.
- `[[gaussian-distribution]]` — gives the Gaussian likelihood behind MSE.
- `[[bayesian-statistics]]` — extends likelihood with priors.
- `[[loss-functions]]` — compares likelihood-based losses with other objectives.
- `[[cross-entropy-nll]]` — deep-learning form of categorical NLL.
- `[[logistic-regression]]` — uses Bernoulli likelihood for classification.

## Resources

- Wasserman, "All of Statistics." Springer, 2004.
- Bishop, "Pattern Recognition and Machine Learning." Springer, 2006.
- Murphy, "Probabilistic Machine Learning: An Introduction." MIT Press, 2022. <https://probml.github.io/pml-book/book1.html>
