# Adam Optimization

## Overview

Adam is the default optimizer for many deep learning workloads because it combines momentum with per-parameter adaptive learning rates. It is robust when gradients have different scales across layers, sparse features, or poorly conditioned curvature.

Adam is not magic. Its behavior depends on learning rate, moment decay rates, epsilon placement, batch size, weight decay, and warmup. Principal-level interviews often test whether you know the update equations and the practical failure modes.

## Math / Derivation

For gradient `g_t = grad_theta L_t(theta)` at step `t`, Adam keeps two exponential moving averages:

```
m_t = beta1 m_{t-1} + (1 - beta1) g_t
v_t = beta2 v_{t-1} + (1 - beta2) g_t^2
```

Because `m_0 = 0` and `v_0 = 0`, the early moments are biased toward zero. Adam corrects this:

```
mhat_t = m_t / (1 - beta1^t)
vhat_t = v_t / (1 - beta2^t)
```

The parameter update is:

```
theta_t = theta_{t-1} - alpha * mhat_t / (sqrt(vhat_t) + epsilon)
```

Typical defaults are:

```
alpha = 1e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
```

AdamW decouples weight decay from the gradient normalization:

```
theta_t = theta_{t-1} - alpha * weight_decay * theta_{t-1}
theta_t = theta_t - alpha * mhat_t / (sqrt(vhat_t) + epsilon)
```

This differs from adding an L2 penalty to the loss, especially under adaptive scaling.

## Intuition

The first moment is momentum: it smooths noisy gradients and keeps moving in directions that are consistently useful. The second moment is a scale tracker: parameters with historically large gradients get smaller effective steps, while parameters with small gradients get larger relative steps.

Bias correction matters because the moment averages start at zero. Without correction, the first updates would use artificially small moment estimates.

## When & Why

Use Adam or AdamW as a strong default for transformers, language models, sparse-gradient problems, and early experimentation. Use warmup when early gradients are unstable or when training large networks with high target learning rates.

Consider SGD with momentum for some vision workloads or convex-ish models when final generalization is more important than fast training loss reduction. Adam can converge quickly while sometimes landing in sharper minima or overfitting without careful weight decay.

Use AdamW instead of Adam with L2 regularization when you want predictable weight decay. This is standard for modern deep learning.

## Implementation

A later implementation pass should implement Adam and AdamW from scratch with immutable tensor updates, validate step-by-step values against PyTorch for a controlled gradient sequence, and show the effect of removing bias correction. It should also compare Adam, AdamW, RMSProp, and SGD with momentum on the same synthetic regression or classification task.

The implementation should test edge cases: zero gradients, sparse large gradients, epsilon sensitivity, and different beta values.

## Cross-links

- `[[gradient-descent]]` — Adam is an adaptive first-order optimizer.
- `[[loss-functions]]` — Adam follows gradients of the chosen objective.
- `[[l1-l2-regularization]]` — AdamW decouples weight decay from L2-style loss penalties.
- `[[linear-regression]]` — convex regression is a clean sandbox for optimizer comparisons.
- `[[vanishing-exploding-gradients]]` — Adam does not fix missing or unstable gradient signal by itself.

## Resources

- Diederik P. Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization." ICLR 2015.
- Ilya Loshchilov and Frank Hutter, "Decoupled Weight Decay Regularization." ICLR 2019.
- Sebastian Ruder, "An overview of gradient descent optimization algorithms." 2016.
