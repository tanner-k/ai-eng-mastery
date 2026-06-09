# Loss Functions

## Overview

A loss function turns model behavior into a scalar objective that an optimizer can minimize. In AI engineering, this choice is not cosmetic: it defines what errors matter, how gradients flow, how outliers are treated, and what probabilistic assumptions the model is making.

Most training loops look like:

```
prediction = model(x)
loss = L(prediction, target)
parameters <- optimizer step using gradient of loss
```

Changing `L` changes the geometry that `[[gradient-descent]]` sees. A regression model trained with MSE learns conditional means and is highly sensitive to large residuals. A classifier trained with cross-entropy learns calibrated class probabilities under a likelihood model. A ranking model trained with a pairwise or margin loss optimizes ordering, not raw prediction accuracy.

## Math / Derivation

For examples `(x_i, y_i)` and model `f_theta`, empirical risk minimization chooses:

```
theta* = argmin_theta (1/n) sum_i l(f_theta(x_i), y_i)
```

Common losses:

| Task | Loss | Formula |
|---|---|---|
| Regression | MSE | `(1/n) sum_i (yhat_i - y_i)^2` |
| Regression | MAE | `(1/n) sum_i |yhat_i - y_i|` |
| Binary classification | Logistic / BCE | `-y log p - (1-y) log(1-p)` |
| Multiclass classification | Cross-entropy | `-sum_k y_k log p_k` |
| Margin classification | Hinge | `max(0, 1 - y f(x))`, `y in {-1, 1}` |

MSE corresponds to maximum likelihood under Gaussian residuals with constant variance. MAE corresponds to Laplace residuals. Cross-entropy is the negative log-likelihood of the categorical distribution predicted by softmax.

The gradient shape matters. For a scalar prediction, MSE has:

```
d/dyhat (yhat - y)^2 = 2(yhat - y)
```

Large residuals create large gradients. MAE has subgradient `sign(yhat - y)`, so very large residuals do not dominate as strongly. Cross-entropy with softmax gives the especially clean logit gradient:

```
dL/dz = p - y
```

where `z` are logits and `p = softmax(z)`.

## Intuition

A loss function is the scoreboard. If the scoreboard rewards the wrong thing, the optimizer will faithfully improve the wrong thing.

MSE says, "large errors are much worse than small errors." This is useful when outliers are real high-cost failures, but dangerous when labels contain occasional corruption. MAE says, "each unit of error has the same price," which is more robust but less smooth. Cross-entropy says, "assign high probability to the true class," so it punishes confident wrong answers much more than uncertain wrong answers.

## When & Why

Use MSE for well-behaved continuous targets where the mean is the right prediction target. Use MAE, Huber, or quantile losses when labels have heavy tails or when median/percentile behavior matters.

Use binary cross-entropy for independent binary labels and categorical cross-entropy for mutually exclusive classes. Prefer logits-based implementations, such as BCE-with-logits or cross-entropy over raw logits, because they are numerically stable.

Use margin losses when relative separation matters more than calibrated probabilities. Use ranking losses when the product requirement is ordering, such as search or recommendation.

Loss choice also interacts with `[[l1-l2-regularization]]`, `[[evaluation-metrics]]`, and `[[bias-variance]]`: the training objective should be a differentiable proxy for the metric and risk profile you actually care about.

## Implementation

A later implementation pass should build a small loss library from scratch in PyTorch tensors. It should implement MSE, MAE, binary cross-entropy with logits, multiclass cross-entropy, hinge, and Huber losses, then validate their values and gradients against `torch.nn.functional` and `torch.autograd`.

The implementation should emphasize numerical stability: stable sigmoid/log-sigmoid forms for BCE, log-sum-exp for multiclass cross-entropy, and clear behavior at non-smooth points for MAE and hinge losses.

## Cross-links

- `[[gradient-descent]]` — optimizers minimize the scalar objective defined by the loss.
- `[[mle-and-nll]]` — many losses are negative log-likelihoods.
- `[[evaluation-metrics]]` — metrics measure success; losses provide differentiable training signals.
- `[[linear-regression]]` — MSE is the canonical linear regression objective.
- `[[logistic-regression]]` — logistic loss is cross-entropy for binary classification.

## Resources

- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 1.
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*, sections on likelihood and risk minimization.
- Goodfellow, Bengio, and Courville, *Deep Learning*, Chapter 5.
