# Backpropagation

## Overview

Backpropagation is reverse-mode automatic differentiation applied to neural networks. It computes gradients of a scalar loss with respect to every parameter by walking backward through the computation graph and repeatedly applying the chain rule.

The reason deep learning is practical is not that derivatives are simple; it is that backprop reuses intermediate results so the full gradient costs only a small constant factor more than the forward pass. Without this dynamic programming structure, training large networks would require perturbing each parameter individually, which is infeasible.

## Math / Derivation

For a feed-forward network:

```text
a0 = x
zk = ak-1 Wk + bk
ak = phi(zk)
L = loss(aK, y)
```

Backprop computes local vector-Jacobian products from the output layer to the input layer. Define `delta_k = dL/dz_k`. For the last layer:

```text
delta_K = dL/da_K * phi'(z_K)
```

For hidden layers:

```text
delta_k = (delta_(k+1) W_(k+1)^T) * phi'(z_k)
```

Parameter gradients follow from the affine local derivatives:

```text
dL/dW_k = a_(k-1)^T delta_k
dL/db_k = sum over batch of delta_k
dL/da_(k-1) = delta_k W_k^T
```

This is the chain rule organized to avoid constructing full Jacobian matrices.

## Intuition

Backprop asks each operation two questions: what did you output in the forward pass, and how does a small change in your input change the final loss? Each layer receives a gradient signal from the layer after it, multiplies by its local derivative, and sends the result to the layer before it.

The algorithm is efficient because each intermediate activation is reused. A layer does not need to know the whole network; it only needs its cached inputs and the upstream gradient.

## When & Why

Backprop is used whenever a differentiable model is trained with gradient-based optimization. It underlies MLPs, CNNs, transformers, diffusion models, and most self-supervised objectives.

The practical concerns are:

- Storing activations for the backward pass consumes memory.
- Incorrect tensor shapes can produce wrong gradients even when operations run.
- Non-differentiable or saturated operations can block gradient flow.
- Long chains of Jacobian products can cause vanishing or exploding gradients.
- In-place mutation can corrupt cached values needed for backward computation.

## Implementation

A later implementation pass should build a tiny reverse-mode autodiff engine or a two-layer MLP with manual backward equations. It should cache forward activations, compute gradients for weights and biases, and compare every manual gradient against `torch.autograd` on synthetic data.

The implementation should also demonstrate that backprop uses vector-Jacobian products rather than full Jacobians, and it should include shape assertions for each gradient tensor.

## Cross-links

- `[[chain-rule]]` — the mathematical rule backprop applies repeatedly.
- `[[gradients-and-jacobians]]` — explains Jacobians and vector-Jacobian products.
- `[[broadcasting-in-nns]]` — bias gradients are reductions over broadcasted axes.
- `[[activations-tanh-relu]]` — activation derivatives control gradient flow.
- `[[vanishing-exploding-gradients]]` — long products of Jacobians can destabilize learning.
- `[[gradient-descent]]` — optimizers consume the gradients produced by backprop.

## Resources

- Rumelhart, Hinton, and Williams, "Learning representations by back-propagating errors." Nature, 1986.
- Goodfellow, Bengio, and Courville, "Deep Learning", Chapter 6. <https://www.deeplearningbook.org/>
- Karpathy, "Micrograd." <https://github.com/karpathy/micrograd>
