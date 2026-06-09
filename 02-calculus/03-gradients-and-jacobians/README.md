# Gradients and Jacobians

## Overview

Gradients and Jacobians organize partial derivatives for high-dimensional functions. A gradient describes how a scalar output changes with respect to many inputs. A Jacobian describes how many outputs change with respect to many inputs.

AI engineering uses both constantly. Loss functions produce scalar objectives whose gradients update parameters. Model layers, logits, embeddings, and transformations are vector-valued, so their local derivative structure is a Jacobian. Backpropagation avoids materializing most large Jacobians, but the math still determines shape, memory, and gradient-flow behavior.

This topic builds on [[derivatives-and-partials]] and [[chain-rule]], and it prepares for [[backpropagation]], [[softmax]], and [[cross-entropy-nll]].

## Math / Derivation

For a scalar function `f: R^d -> R`, the gradient is

```text
grad f(x) = [partial f / partial x_1, ..., partial f / partial x_d]^T
```

It points in the direction of steepest local increase under the Euclidean norm. The directional derivative in direction `u` is

```text
D_u f(x) = grad f(x)^T u
```

For a vector function `F: R^n -> R^m`, the Jacobian is the matrix of all output-input partials:

```text
J_F(x)[i, j] = partial F_i / partial x_j
```

so `J_F(x)` has shape `m x n`.

Example:

```text
F(x1, x2) = [x1^2 + x2, x1 x2]
```

Then

```text
J_F = [[2x1, 1],
       [x2,  x1]]
```

For a scalar loss `L(y)` and `y = F(x)`, reverse-mode uses:

```text
dL/dx = (dL/dy) J_F
```

depending on row/column convention. The key shape rule is that output sensitivities contract with the Jacobian to produce input sensitivities.

## Intuition

A gradient is a compass for a scalar landscape: it points uphill, and its negative points downhill. Each component says how sensitive the scalar is to one coordinate.

A Jacobian is a local linear map. Near `x`, a small input perturbation `dx` changes the output approximately by:

```text
dF approx J_F dx
```

This local linear map can stretch, shrink, rotate, or mix directions. In deep networks, many local Jacobians are chained together. Their singular values explain whether gradients shrink, explode, or preserve signal.

Most practical systems never build full Jacobian matrices for large layers because they would be too expensive. They compute Jacobian-vector products or vector-Jacobian products instead.

## When & Why

Use gradients when optimizing scalar losses, interpreting parameter sensitivity, or debugging update directions. Use Jacobian reasoning when outputs are vector-valued or when you need to understand how perturbations move through a model.

Common AI engineering examples:

- Gradient of cross-entropy with respect to logits.
- Jacobian of softmax, which couples output probabilities.
- Jacobian-vector products for influence, curvature, and implicit differentiation.
- Vector-Jacobian products in reverse-mode backpropagation.
- Singular values of layer Jacobians in [[vanishing-exploding-gradients]] analysis.

## Implementation

A later implementation pass should compare explicit Jacobians for small functions with PyTorch's automatic differentiation utilities. It should include examples where building the full Jacobian is reasonable, then contrast that with vector-Jacobian and Jacobian-vector products that scale better.

The implementation should validate:

1. Gradient shapes for scalar-valued functions.
2. Jacobian shapes for vector-valued functions.
3. Agreement between manual Jacobians and autograd on small examples.
4. Equivalent results from full Jacobian multiplication and VJP/JVP products.
5. Memory implications of full Jacobians for batched model outputs.

## Cross-links

- `[[derivatives-and-partials]]` — defines the partial derivatives that fill gradients and Jacobians.
- `[[chain-rule]]` — composes Jacobians through computation graphs.
- `[[backpropagation]]` — computes vector-Jacobian products efficiently.
- `[[softmax]]` — has a dense Jacobian because probabilities compete.
- `[[vanishing-exploding-gradients]]` — depends on products of Jacobians across depth.

## Resources

- Matrix Cookbook, "Derivatives." <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey." JMLR 2018. <https://jmlr.org/papers/v18/17-468.html>
- Justin Domke, "Generic Methods for Optimization-Based Modeling." AISTATS 2012. <https://proceedings.mlr.press/v22/domke12.html>
