# Activations: tanh and ReLU

## Overview

Activation functions introduce nonlinearity into neural networks. Without them, a stack of linear layers collapses into one linear transformation no matter how many layers it has. `tanh` and ReLU are two foundational activations that illustrate the main tradeoff: smooth bounded outputs versus sparse, piecewise-linear outputs.

AI engineers need to understand activations because they control representational power, gradient flow, initialization sensitivity, sparsity, and numerical stability.

## Math / Derivation

The hyperbolic tangent is:

```text
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
d/dx tanh(x) = 1 - tanh(x)^2
```

It maps real inputs to `(-1, 1)` and is zero-centered.

ReLU is:

```text
relu(x) = max(0, x)
d/dx relu(x) = 1 if x > 0, 0 if x < 0
```

At `x = 0`, the derivative is undefined mathematically; frameworks choose a subgradient, commonly `0`.

For a layer `a = phi(z)`, backprop multiplies upstream gradient by `phi'(z)`. If `phi'(z)` is near zero for many units, earlier layers receive little gradient signal.

## Intuition

`tanh` squashes large positive and negative values into a bounded range. This can be useful for centered hidden states, but saturation makes gradients vanish at large magnitudes.

ReLU keeps positive values unchanged and shuts off negative values. Positive activations preserve gradient magnitude, making deep networks easier to train than with saturated sigmoids or tanh. The cost is dead units: if a ReLU unit stays negative for all inputs, its gradient is zero and it may stop learning.

## When & Why

ReLU and variants are default choices for many feed-forward and convolutional networks because they are cheap, sparse, and reduce vanishing gradients on active paths. `tanh` is still useful when bounded, zero-centered outputs are part of the design, such as some recurrent-state transformations or output constraints.

Activation choice interacts with initialization. ReLU networks commonly use He initialization; tanh networks often use Xavier/Glorot initialization to keep activation variance stable.

## Implementation

A later implementation pass should implement `tanh`, ReLU, and their backward functions manually on synthetic tensors. It should compare gradients to PyTorch, plot or print saturation/dead-unit statistics, and train a small MLP with tanh versus ReLU to show different gradient-flow behavior.

The implementation should frame activation derivatives as local factors in backprop rather than standalone training algorithms.

## Cross-links

- `[[backpropagation]]` — activation derivatives gate upstream gradients.
- `[[vanishing-exploding-gradients]]` — saturated activations contribute to vanishing gradients.
- `[[gradient-descent]]` — optimizer behavior depends on gradients produced through activations.
- `[[batch-normalization]]` — normalization changes the distribution entering activations.
- `[[cnns]]` — ReLU-style activations are common after convolutional layers.

## Resources

- Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural networks." AISTATS 2010.
- He et al., "Delving Deep into Rectifiers." ICCV 2015. <https://arxiv.org/abs/1502.01852>
- PyTorch documentation, activation functions. <https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions>
