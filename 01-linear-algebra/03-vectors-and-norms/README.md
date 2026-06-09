# Vectors and Norms

## Overview

Vectors are the basic representation unit for modern AI systems: embeddings, model weights, gradients, logits, hidden states, and feature rows are all vectors. Norms measure vector size, distances measure separation, and dot products or cosine similarity measure alignment.

A principal AI engineer uses vector geometry constantly: diagnosing exploding gradients, choosing similarity metrics for retrieval, understanding regularization, interpreting normalization layers, and reasoning about whether two representations carry similar information. The notation is simple, but the engineering consequences show up in model stability and product behavior.

## Math / Derivation

For x in R^d, common vector norms include:

```
L1 norm:      ||x||_1 = sum_i |x_i|
L2 norm:      ||x||_2 = sqrt(sum_i x_i^2)
L-infinity:   ||x||_inf = max_i |x_i|
```

A norm must satisfy:

1. Non-negativity: `||x|| >= 0`, and `||x|| = 0` only when x = 0.
2. Absolute homogeneity: `||a x|| = |a| ||x||`.
3. Triangle inequality: `||x + y|| <= ||x|| + ||y||`.

The dot product is:

```
x dot y = sum_i x_i y_i
```

It connects algebra to geometry:

```
x dot y = ||x||_2 ||y||_2 cos(theta)
```

where theta is the angle between x and y. Cosine similarity divides out vector length:

```
cos_sim(x, y) = (x dot y) / (||x||_2 ||y||_2)
```

Euclidean distance is the L2 norm of a difference:

```
dist_2(x, y) = ||x - y||_2
```

For differentiable points where x != 0:

```
grad_x ||x||_2 = x / ||x||_2
grad_x (1/2 ||x||_2^2) = x
```

The squared norm has a simpler gradient and is often used in losses and penalties for that reason.

## Intuition

The L2 norm is ordinary geometric length. Scaling a vector by 3 triples its L2 norm. The L1 norm measures total absolute mass and tends to emphasize sparsity because moving weight onto fewer coordinates can be cheaper under L1-regularized objectives. The L-infinity norm asks only for the largest absolute coordinate, which is useful when the worst individual component matters.

Cosine similarity asks whether two vectors point in the same direction, not whether they have the same magnitude. This is why cosine is common for text embeddings: a long document embedding and a short query embedding may have different lengths, but their direction can still capture semantic alignment.

## When & Why

Use L2 distance when magnitude and direction both matter, such as ordinary Euclidean geometry or least-squares residuals. Use cosine similarity when direction should dominate magnitude, especially in embedding retrieval. Use L1 penalties when sparsity or feature selection is desired, and L2 penalties when smooth shrinkage is preferred.

Norms are also operational metrics. Gradient norm spikes often indicate instability before loss becomes `nan`. Weight norms help diagnose regularization strength. Activation norms reveal saturation, collapse, or distribution shift. In high-dimensional systems, raw intuition from 2D geometry can mislead, so metrics should be checked empirically.

## Implementation

A later implementation pass should build vector and norm utilities with PyTorch and validate them against `torch.linalg.vector_norm`, `torch.nn.functional.cosine_similarity`, and autograd. It should validate:

1. L1, L2, and L-infinity norm calculations.
2. Pairwise Euclidean distance and cosine similarity for batches of vectors.
3. Manual gradients for L2 norm and squared L2 norm.
4. Numerical stability for near-zero vectors using epsilon terms.
5. Retrieval-ranking differences between dot product, cosine similarity, and Euclidean distance.

The implementation should emphasize where a metric changes model behavior, not just how to compute the formula.

## Cross-links

- `[[matrix-multiplication]]` - matrix products are collections of vector dot products.
- `[[broadcasting]]` - pairwise distance and normalization code often relies on broadcasting.
- `[[l1-l2-regularization]]` - regularizers are norm penalties on parameters.
- `[[evaluation-metrics]]` - ranking and distance metrics determine retrieval quality.
- `[[vanishing-exploding-gradients]]` - gradient norms are core diagnostic signals.

## Resources

- Gilbert Strang, *Introduction to Linear Algebra*, Wellesley-Cambridge Press.
- Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, Chapter 3: <https://web.stanford.edu/~boyd/cvxbook/>
- PyTorch `torch.linalg.vector_norm` documentation: <https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html>
