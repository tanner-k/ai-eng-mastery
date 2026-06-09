# Broadcasting

## Overview

Broadcasting is the set of rules that lets array libraries apply elementwise operations to tensors with different but compatible shapes. It is why a bias vector of shape `(hidden,)` can be added to an activation matrix of shape `(batch, hidden)`, why normalization statistics can be applied across a batch, and why masks can be expanded across attention heads without explicit copies.

In AI engineering, broadcasting is both a productivity feature and a common source of subtle bugs. Correct broadcasting removes slow, memory-heavy manual expansion. Incorrect broadcasting can silently apply a value along the wrong axis and produce plausible-looking but wrong results.

## Math / Derivation

Broadcasting compares shapes from right to left. Two dimensions are compatible when:

1. They are equal, or
2. One of them is 1.

Missing leading dimensions behave as if they were 1. The broadcasted output shape uses the maximum size along each aligned dimension.

For example:

```
X shape:      (32, 128)
b shape:          (128)
result shape: (32, 128)
```

The bias is treated as if it had shape `(1, 128)` and is reused across 32 rows.

For a more explicit example:

```
A shape: (4, 1, 7)
B shape: (1, 3, 7)
C shape: (4, 3, 7)
```

The mathematical value of the result is:

```
C[i, j, k] = A[i, 0, k] + B[0, j, k]
```

Broadcasting does not mean data must be physically copied. Libraries can represent expanded dimensions with strides, so the same memory value is read many times. During backpropagation, gradients through broadcasted dimensions must be reduced by summing along every axis that was expanded.

If:

```
Y = X + b
X shape = (n, d)
b shape = (d,)
```

and G = dL/dY, then:

```
dL/dX = G
dL/db = sum over rows of G
```

The reduction appears because each element of b influenced n output elements.

## Intuition

Broadcasting is "pretend repeat without actually copying." A scalar can act on every element. A row vector can act on every row. A `(batch, 1)` column can scale every feature in each example. The convenience is powerful because it lets code express the intended mathematical relationship without manual loops.

The danger is that "compatible" is not the same as "intended." A tensor with shape `(batch, 1)` and a tensor with shape `(features,)` can combine into `(batch, features)`. That may be exactly right for per-example scaling, or completely wrong if the missing dimension was accidental.

## When & Why

Broadcasting appears in nearly every model:

- Bias addition after [[matrix-multiplication]].
- Feature normalization with per-feature means and variances.
- Attention masks expanded over batch or head dimensions.
- Loss weights applied per example, per class, or per token.
- Distance computations such as pairwise differences between two sets of vectors.

Use broadcasting when the repeated value has a clear semantic axis. Avoid relying on it when shape names are ambiguous. In production training code, strategically placed shape assertions are often worth more than clever compact expressions.

## Implementation

A later implementation pass should build a small broadcasting explainer using PyTorch tensors. It should not reimplement PyTorch internals, but it should validate:

1. A pure shape-inference function for broadcast compatibility.
2. Elementwise operations with scalars, vectors, matrices, and rank-3 tensors.
3. Gradient reductions over broadcasted dimensions compared against `torch.autograd`.
4. Memory behavior differences between `expand` and `repeat`.
5. Failure cases where shapes are compatible but semantically suspicious.

The implementation should teach the rule and its consequences, especially the backward-pass summation over expanded axes.

## Cross-links

- `[[matrix-multiplication]]` - linear-layer bias addition uses broadcasting.
- `[[vectors-and-norms]]` - normalization often broadcasts vector norms or statistics.
- `[[broadcasting-in-nns]]` - deep learning layers use broadcasting for masks, biases, and normalization.
- `[[softmax]]` - stable softmax subtracts a broadcasted maximum.
- `[[batch-normalization]]` - batch statistics are broadcast over examples and spatial axes.

## Resources

- NumPy broadcasting documentation: <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- PyTorch broadcasting semantics: <https://pytorch.org/docs/stable/notes/broadcasting.html>
- PyTorch `expand` documentation: <https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html>
