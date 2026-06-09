# Bias-Variance Tradeoff

## Overview

The bias-variance tradeoff explains why models fail to generalize. Bias is systematic error from assumptions that are too simple or wrong. Variance is sensitivity to the particular training sample. Good AI engineering balances both: a model must be expressive enough to capture the signal but stable enough not to chase noise.

This topic connects [[probability-basics]], [[mle-and-nll]], [[bayesian-statistics]], [[evaluation-metrics]], [[l1-l2-regularization]], and tree ensembles such as [[random-forests-xgboost]].

## Math / Derivation

For supervised regression, assume:

```text
y = f(x) + epsilon
E[epsilon] = 0
Var(epsilon) = sigma^2
```

Let `f_hat(x)` be the model learned from a random training set. The expected squared prediction error at `x` decomposes as:

```text
E[(y - f_hat(x))^2]
  = Bias[f_hat(x)]^2 + Var[f_hat(x)] + sigma^2
```

where:

```text
Bias[f_hat(x)] = E[f_hat(x)] - f(x)
Var[f_hat(x)] = E[(f_hat(x) - E[f_hat(x)])^2]
```

The noise term `sigma^2` is irreducible error. More data, better features, or better models can reduce bias and variance, but they cannot remove inherent label noise.

## Intuition

High bias means the model is consistently wrong in the same way. A linear model on a strongly nonlinear relationship may underfit every training sample.

High variance means the model changes too much depending on the data it saw. A deep decision tree can fit one sample nearly perfectly but behave differently on another sample from the same population.

Regularization, ensembling, data augmentation, and early stopping are variance-control tools. More expressive models and better features reduce bias, but they can increase variance if not constrained.

## When & Why

Use bias-variance reasoning when diagnosing train/test gaps:

- High train error and high validation error usually indicate high bias or optimization failure.
- Low train error and high validation error usually indicate high variance or data leakage.
- More data usually reduces variance more than bias.
- More model capacity usually reduces bias but may increase variance.
- Ensembling often reduces variance without increasing bias much.

The tradeoff is not a fixed law that every modern model follows monotonically. Overparameterized neural networks can show double descent, but bias-variance remains a useful diagnostic lens.

## Implementation

A later implementation pass should simulate repeated training sets from a known function and fit models with different capacity. It should estimate empirical bias and variance at fixed test points by retraining many times.

The implementation should validate the decomposition for squared error, compare underfit and overfit models, and show how ensembling reduces variance.

## Cross-links

- `[[probability-basics]]` — frames training data and metrics as random samples.
- `[[bayesian-statistics]]` — priors can reduce variance by constraining estimates.
- `[[evaluation-metrics]]` — validation metrics reveal generalization behavior.
- `[[l1-l2-regularization]]` — controls variance by penalizing model complexity.
- `[[decision-trees]]` — high-variance models when grown deeply.
- `[[random-forests-xgboost]]` — ensembles reduce variance and manage bias.

## Resources

- Hastie, Tibshirani, and Friedman, "The Elements of Statistical Learning." <https://hastie.su.domains/ElemStatLearn/>
- Geman, Bienenstock, and Doursat, "Neural Networks and the Bias/Variance Dilemma." Neural Computation, 1992.
- Belkin et al., "Reconciling modern machine-learning practice and the classical bias-variance trade-off." PNAS, 2019. <https://www.pnas.org/doi/10.1073/pnas.1903070116>
