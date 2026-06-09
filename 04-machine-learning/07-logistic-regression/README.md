# Logistic Regression

## Overview

Logistic regression is a linear classifier that predicts probabilities. It maps a linear score, or logit, through the sigmoid function:

```
z = x^T w + b
p(y=1 | x) = sigmoid(z) = 1 / (1 + exp(-z))
```

It remains a core AI engineering model because it is fast, interpretable, calibrated more easily than many nonlinear models, and an essential baseline for classification.

## Math / Derivation

For binary labels `y in {0, 1}`, the Bernoulli likelihood is:

```
p(y | x) = p^y (1-p)^(1-y)
```

The negative log-likelihood over examples is binary cross-entropy:

```
L(w, b) = -(1/n) sum_i [y_i log p_i + (1-y_i) log(1-p_i)]
```

With logits `z_i = x_i^T w + b` and `p_i = sigmoid(z_i)`, the gradient is:

```
grad_w L = (1/n) X^T (p - y)
grad_b L = (1/n) sum_i (p_i - y_i)
```

The decision boundary at threshold `0.5` is:

```
x^T w + b = 0
```

Changing the probability threshold changes the operating point but not the learned linear score.

## Intuition

Linear regression predicts unbounded values. Logistic regression predicts log-odds:

```
log(p / (1-p)) = x^T w + b
```

A one-unit increase in feature `j` changes the log-odds by `w_j`, holding other features fixed. Positive weights push probability up; negative weights push it down.

## When & Why

Use logistic regression for binary classification baselines, sparse high-dimensional features, interpretable risk models, and calibrated scoring systems. It is especially strong when features are already informative and interactions are limited or engineered.

It struggles with nonlinear decision boundaries unless features are transformed. It can also look deceptively good under class imbalance if evaluated only with accuracy; use `[[evaluation-metrics]]` such as precision, recall, PR-AUC, and calibration.

Regularization is usually necessary for high-dimensional data and separable classes. Without it, coefficients can grow extremely large when the data is linearly separable.

## Implementation

A later implementation pass should implement binary logistic regression from scratch with stable BCE-with-logits, analytic gradients, and optional L1/L2 regularization. It should validate gradients against `torch.autograd`, compare probability thresholds, and report accuracy, precision, recall, F1, ROC-AUC, and calibration.

The implementation should avoid computing `log(sigmoid(z))` directly; stable log-sigmoid or BCE-with-logits algebra should be used.

## Cross-links

- `[[loss-functions]]` — logistic regression minimizes binary cross-entropy.
- `[[mle-and-nll]]` — BCE is the Bernoulli negative log-likelihood.
- `[[l1-l2-regularization]]` — penalties control coefficient growth.
- `[[evaluation-metrics]]` — thresholds define precision/recall tradeoffs.
- `[[linear-regression]]` — logistic regression is linear in log-odds.

## Resources

- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, logistic regression sections.
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, generalized linear models.
- scikit-learn User Guide, "Logistic regression."
