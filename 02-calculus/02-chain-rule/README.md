# The Chain Rule

## Overview

The chain rule explains how local changes propagate through composed functions. In AI systems, nearly every model is a composition: embeddings feed layers, layers feed activations, activations feed logits, logits feed losses. The derivative of the whole system is built by multiplying the derivatives of the pieces.

Backpropagation is the chain rule organized as a dynamic program over a computation graph. Instead of expanding a huge symbolic expression, autograd systems store local operations during the forward pass and traverse them backward to accumulate derivatives.

This topic connects [[derivatives-and-partials]] to [[gradients-and-jacobians]], [[gradient-descent]], and [[backpropagation]].

## Math / Derivation

For single-variable functions `y = f(u)` and `u = g(x)`, the derivative of the composition is

```text
d/dx f(g(x)) = f'(g(x)) g'(x)
```

For a scalar loss with intermediate variables,

```text
x -> u -> v -> L
```

the total sensitivity is the product of local sensitivities:

```text
dL/dx = (dL/dv)(dv/du)(du/dx)
```

For vector-valued intermediates, the same idea uses Jacobians. If `z = g(x)` and `y = f(z)`, then

```text
J_{y,x} = J_{y,z} J_{z,x}
```

For scalar losses, reverse-mode autodiff works with vector-Jacobian products:

```text
dL/dx = (dL/dz) J_{z,x}
```

Example:

```text
L = (sigma(wx + b) - y)^2
a = wx + b
p = sigma(a)
L = (p - y)^2
```

Then

```text
dL/dw = 2(p - y) sigma(a)(1 - sigma(a)) x
dL/db = 2(p - y) sigma(a)(1 - sigma(a))
```

Each factor comes from one local edge in the computation graph.

## Intuition

The chain rule is responsibility passing. If the loss changes when `p` changes, and `p` changes when `a` changes, and `a` changes when `w` changes, then `w` inherits responsibility through the product of those links.

Multiplication is the important detail. If any link has near-zero derivative, the upstream signal shrinks. If many links have derivative magnitude greater than one, the upstream signal can explode. This is the calculus behind vanishing and exploding gradients.

Autograd does not need to know about "neural networks" specifically. It only needs local derivative rules for primitive operations and a graph describing how outputs were composed from inputs.

## When & Why

Use the chain rule whenever outputs are composed from intermediate computations:

- Deriving gradients for logistic regression, cross-entropy, and softmax.
- Understanding [[backpropagation]] in multilayer networks.
- Debugging saturated activations where sigmoid or tanh derivatives are near zero.
- Reasoning about gradient flow through normalization, residual connections, and attention.

The most common error is dropping a factor. If a derivation seems too simple for a composed model, check whether every intermediate variable contributed its local derivative.

## Implementation

A later implementation pass should build a tiny reverse-mode autodiff engine for scalar expressions. It should support primitive operations such as addition, multiplication, exponentiation, `tanh`, and `sigmoid`, then compare its gradients with PyTorch autograd.

The implementation should validate:

1. Local derivative rules for each primitive.
2. Correct gradient accumulation when a value is reused in multiple branches.
3. Manual chain-rule derivations for nested scalar functions.
4. Agreement with PyTorch on small composed expressions.

## Cross-links

- `[[derivatives-and-partials]]` — provides the local derivative rules used by the chain rule.
- `[[gradients-and-jacobians]]` — generalizes the chain rule to vector-valued functions.
- `[[backpropagation]]` — implements reverse-mode chain rule for neural networks.
- `[[vanishing-exploding-gradients]]` — arises from repeated multiplication of local derivatives.

## Resources

- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey." JMLR 2018. <https://jmlr.org/papers/v18/17-468.html>
- Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation." <https://karpathy.ai/zero-to-hero.html>
- Gilbert Strang, "Calculus." MIT OpenCourseWare. <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/>
