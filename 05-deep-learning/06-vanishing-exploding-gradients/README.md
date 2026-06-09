# Vanishing and Exploding Gradients

## Overview

Vanishing and exploding gradients are training pathologies caused by repeatedly multiplying derivatives through many layers or time steps. If those factors are usually smaller than one, early-layer gradients shrink toward zero. If they are usually larger than one, gradients grow until updates become unstable or numerical values overflow.

This topic matters because deep networks can fail even when the loss, optimizer, and data are reasonable. The architecture may simply not allow useful gradient signal to reach the parameters that need it.

## Math / Derivation

For a chain of functions:

```text
h_L = f_L(f_(L-1)(...f_1(x)))
```

the gradient with respect to an early hidden state includes a product of Jacobians:

```text
dL/dh_k = dL/dh_L * J_L * J_(L-1) * ... * J_(k+1)
```

Norms can be bounded by:

```text
||dL/dh_k|| <= ||dL/dh_L|| * product_i ||J_i||
```

If typical Jacobian norms are less than `1`, the product decays exponentially with depth. If they are greater than `1`, it grows exponentially.

For scalar recurrence `h_t = w h_(t-1)`, the derivative is:

```text
dh_T/dh_0 = w^T
```

This shows the core mechanism directly.

## Intuition

Backpropagation sends a signal backward through the network. Each layer can dampen or amplify that signal. A hundred mild dampening factors become near-zero when multiplied. A hundred mild amplifications can become enormous.

The problem is not only numerical. Vanishing gradients mean early layers do not learn useful features. Exploding gradients mean updates are dominated by unstable steps rather than meaningful descent.

## When & Why

These pathologies are common in very deep feed-forward networks, recurrent networks over long sequences, poorly initialized models, saturated activations, and systems trained with overly large learning rates.

Common mitigations include:

- Better initialization such as Xavier or He initialization.
- ReLU-style activations on active paths.
- Residual connections that provide shorter gradient routes.
- Normalization layers.
- Gradient clipping for explosions.
- Gated recurrent architectures for long sequences.

## Implementation

A later implementation pass should create synthetic deep MLPs or scalar recurrent chains that measure gradient norm by depth under different initializations and activations. It should demonstrate both vanishing and exploding behavior, then show how clipping, initialization, normalization, or residual connections changes the gradient profile.

The implementation should report gradient norms rather than only final losses, because the point is to diagnose signal propagation.

## Cross-links

- `[[backpropagation]]` — the pathologies arise from chained derivatives.
- `[[activations-tanh-relu]]` — saturation and ReLU activity affect gradient flow.
- `[[batch-normalization]]` — normalization can stabilize activation distributions.
- `[[gradient-descent]]` — exploding gradients produce unstable optimizer steps.
- `[[vectors-and-norms]]` — gradient norms are the primary diagnostic.

## Resources

- Hochreiter, "Untersuchungen zu dynamischen neuronalen Netzen." 1991.
- Pascanu, Mikolov, and Bengio, "On the difficulty of training recurrent neural networks." ICML 2013. <https://arxiv.org/abs/1211.5063>
- He et al., "Deep Residual Learning for Image Recognition." CVPR 2016. <https://arxiv.org/abs/1512.03385>
