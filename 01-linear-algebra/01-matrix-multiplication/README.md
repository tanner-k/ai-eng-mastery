# Matrix Multiplication

## Overview

Matrix multiplication is the algebraic operation behind linear layers, attention projections, convolutions lowered to matrix products, embedding lookups followed by projections, and most GPU kernels used in model training. If a model maps a batch of feature vectors to logits, hidden states, or attention scores, the central operation is usually a product such as `XW`, `QK^T`, or `A @ B`.

For AI engineers, matrix multiplication is not just notation. It determines tensor shapes, parameter counts, memory traffic, numerical cost, and gradient formulas. A weak grasp of it shows up quickly as silent shape bugs, inefficient loops, or incorrect reasoning about why a model layer has the dimensions it does.

## Math / Derivation

For matrices A in R^(m x n) and B in R^(n x p), the product C = AB is defined only when the inner dimensions match. The result C is in R^(m x p), with entries:

```
C_ij = sum_{k=1}^n A_ik B_kj
```

Each output entry is the dot product of one row of A and one column of B. The operation costs O(mnp) multiply-adds in the dense case.

Matrix multiplication is associative:

```
(AB)C = A(BC)
```

when all shapes are compatible, but it is not generally commutative:

```
AB != BA
```

and sometimes only one of the two products is even defined.

For a linear layer with input batch X in R^(b x d_in), weights W in R^(d_in x d_out), and bias b in R^(d_out), predictions are:

```
Y = XW + b
```

The bias addition relies on [[broadcasting]]. Given an upstream gradient G = dL/dY in R^(b x d_out), the core reverse-mode gradients are:

```
dL/dX = G W^T
dL/dW = X^T G
dL/db = sum over batch rows of G
```

These formulas are why transposes appear constantly in neural network backpropagation.

## Intuition

You can read A @ B as "apply every column-shaped feature mixer in B to every row-shaped example in A." If X is a batch of examples and W is a bank of output directions, then each output cell says how strongly one example aligns with one learned direction.

Another useful view is composition. A matrix represents a linear transformation: rotate, scale, shear, project, or mix coordinates. Multiplying matrices composes transformations. If z = (AB)x, then B acts on x first and A acts next. This "rightmost first" ordering is a common source of confusion when reading model equations.

## When & Why

Use matrix multiplication when many independent dot products share the same operands. A Python loop over examples or output units hides the real structure and prevents hardware from using optimized BLAS or GPU kernels.

Shape reasoning is often the fastest way to debug model code. For example, attention scores use QK^T because Q and K both have shape `(batch, heads, tokens, d_head)`, and the desired score matrix is `(batch, heads, query_tokens, key_tokens)`. Multiplying Q by K without transposing the last two dimensions would be a shape error or the wrong computation.

Multiplication order also matters for performance. Because `(AB)C` and `A(BC)` produce the same mathematical result but can require different intermediate sizes, principal engineers should reason about both correctness and memory movement. In large models, the bottleneck is often not the number of floating-point operations alone but whether intermediates fit in fast memory.

## Implementation

A later implementation pass should build matrix multiplication from first principles using PyTorch tensor indexing or simple loops for clarity, then compare it to `torch.matmul` and the `@` operator. That implementation should validate:

1. Shape compatibility and output shape calculation.
2. Numerical equality against PyTorch for vectors, matrices, and batched matrices.
3. Manual gradients for `Y = XW + b` against `torch.autograd`.
4. The performance gap between naive loops and vectorized matmul.
5. Common failure modes such as swapped dimensions and accidental elementwise multiplication.

The implementation should be framed as an educational reference, not a replacement for optimized PyTorch kernels.

## Cross-links

- `[[broadcasting]]` - bias addition and batched matmul depend on broadcast semantics.
- `[[vectors-and-norms]]` - each matrix product entry is a dot product between vectors.
- `[[eigendecomposition-svd]]` - decompositions explain how matrices scale and rotate special directions.
- `[[linear-regression]]` - linear models are usually written as XW + b.
- `[[backpropagation]]` - reverse-mode gradients for matrix products drive neural network training.

## Resources

- Gilbert Strang, *Introduction to Linear Algebra*, Wellesley-Cambridge Press.
- Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, Johns Hopkins University Press.
- PyTorch documentation, `torch.matmul`: <https://pytorch.org/docs/stable/generated/torch.matmul.html>
