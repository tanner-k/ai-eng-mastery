# Eigendecomposition and SVD

## Overview

Eigendecomposition and singular value decomposition (SVD) reveal the directions along which a linear transformation has its simplest behavior. They explain rank, conditioning, dimensionality reduction, PCA, low-rank approximation, covariance structure, and why gradients can vanish or explode through repeated linear maps.

For AI engineers, these tools show up both directly and indirectly. PCA uses eigenvectors or SVD. Embedding compression and adapter methods rely on low-rank structure. Optimizer and training diagnostics often refer to spectra, singular values, and condition numbers. A principal-level understanding lets you reason about the geometry of a model rather than treating matrices as opaque parameter blocks.

## Math / Derivation

For a square matrix A in R^(n x n), an eigenvector v and eigenvalue lambda satisfy:

```
A v = lambda v
```

The vector v keeps its direction under A; it is only scaled by lambda. If A has n linearly independent eigenvectors, it can be diagonalized:

```
A = V Lambda V^(-1)
```

where columns of V are eigenvectors and Lambda is diagonal with eigenvalues.

For real symmetric matrices, the structure is especially clean:

```
A = Q Lambda Q^T
```

where Q is orthonormal. Covariance matrices are symmetric positive semidefinite, so their eigenvalues are nonnegative and their eigenvectors define principal directions of variance.

SVD applies to any real matrix A in R^(m x n):

```
A = U Sigma V^T
```

where U has orthonormal columns, V has orthonormal columns, and Sigma contains nonnegative singular values sorted from largest to smallest. The singular values are the square roots of the eigenvalues of `A^T A`:

```
A^T A = V Sigma^2 V^T
```

The rank of A is the number of nonzero singular values. The condition number is:

```
kappa(A) = sigma_max / sigma_min
```

for full-rank A. Large condition numbers indicate that some directions are amplified much more than others.

The best rank-k approximation in Frobenius norm is obtained by truncating the SVD:

```
A_k = U_k Sigma_k V_k^T
```

This is the Eckart-Young theorem and is the mathematical basis for PCA and many compression methods.

## Intuition

Eigendecomposition asks: "Are there directions this transformation simply stretches or flips?" SVD asks a more general question: "Which input directions are transformed into which output directions, and by how much?" SVD works even when the matrix is rectangular, which is why it is the more broadly useful tool in ML systems.

Singular values describe axis lengths after the matrix maps the unit sphere into an ellipsoid. A large singular value stretches one direction strongly. A tiny singular value nearly collapses one direction. If many singular values are near zero, the matrix is effectively low-rank and loses information.

## When & Why

Use eigendecomposition for square operators with special structure, especially symmetric matrices such as covariance, Hessian approximations, graph Laplacians, and kernel matrices. Use SVD when the matrix is rectangular, noisy, or you care about rank, conditioning, or low-rank approximation.

In model work, SVD helps answer practical questions:

- Is this embedding matrix effectively low-rank?
- How much information does a rank-k approximation preserve?
- Are features collinear or ill-conditioned?
- Why does repeated multiplication amplify or suppress signals?
- Can a large dense matrix be approximated by smaller factors?

These questions connect directly to [[pca]], training stability, and representation compression.

## Implementation

A later implementation pass should use PyTorch to explore eigendecomposition and SVD numerically rather than reimplementing production-grade decomposition algorithms. It should validate:

1. Eigenpairs for small hand-checkable matrices.
2. Reconstruction error for `A = U @ diag(S) @ V.T`.
3. Relationship between singular values of A and eigenvalues of `A.T @ A`.
4. Low-rank approximation error as k increases.
5. PCA on synthetic correlated data using centered data and SVD.
6. Conditioning effects on solving least-squares problems.

The implementation should explain algorithmic limits: real systems call optimized LAPACK/cuSOLVER routines rather than using naive decomposition code.

## Cross-links

- `[[matrix-multiplication]]` - decompositions factor matrices into structured products.
- `[[vectors-and-norms]]` - eigenvectors and singular vectors are special directions with norm constraints.
- `[[pca]]` - PCA is SVD or eigendecomposition applied to centered data or covariance.
- `[[linear-regression]]` - rank and conditioning affect least-squares solutions.
- `[[vanishing-exploding-gradients]]` - singular values of Jacobians explain gradient scaling through depth.

## Resources

- Gilbert Strang, *Linear Algebra and Learning from Data*, Wellesley-Cambridge Press.
- Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, Johns Hopkins University Press.
- Eckart and Young, "The approximation of one matrix by another of lower rank", Psychometrika, 1936.
- PyTorch `torch.linalg.svd` documentation: <https://pytorch.org/docs/stable/generated/torch.linalg.svd.html>
