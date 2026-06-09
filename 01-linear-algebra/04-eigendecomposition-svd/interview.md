# Eigendecomposition and SVD — Interview Prep

## Q&A

1. **Q: What is an eigenvector?**
   **A:** An eigenvector is a nonzero vector whose direction is unchanged by a square matrix transformation. The matrix only scales it by an eigenvalue: `Av = lambda v`.

2. **Q: When does eigendecomposition apply cleanly?**
   **A:** It applies to square matrices, and it is especially well-behaved for real symmetric matrices, which have orthonormal eigenvectors and real eigenvalues.

3. **Q: Why is SVD more general than eigendecomposition?**
   **A:** SVD applies to any rectangular or square matrix. It decomposes A into input directions, output directions, and nonnegative singular values.

4. **Q: What do singular values mean geometrically?**
   **A:** They are the stretch factors of the linear map along orthogonal input directions. The unit sphere maps to an ellipsoid whose axis lengths are singular values.

5. **Q: How is SVD related to PCA?**
   **A:** PCA finds directions of maximum variance in centered data. Those directions are the right singular vectors of the centered data matrix, equivalently eigenvectors of the covariance matrix.

6. **Q: What does matrix rank mean in terms of singular values?**
   **A:** Rank is the number of nonzero singular values. Small near-zero singular values indicate approximate low-rank structure.

7. **Q: What is the condition number, and why does it matter?**
   **A:** The condition number is `sigma_max / sigma_min` for a full-rank matrix. A large value means the matrix stretches some directions far more than others, making inverse or least-squares problems sensitive to noise.

8. **Q: Why is truncated SVD useful?**
   **A:** Keeping only the top k singular values gives the best rank-k approximation in Frobenius norm. This supports compression, denoising, and dimensionality reduction.

9. **Q: How do singular values relate to vanishing or exploding gradients?**
   **A:** Backpropagation multiplies by Jacobians. If their singular values are mostly below 1, gradients shrink through depth; if above 1, gradients can grow.

10. **Q: Why not implement SVD from scratch in production model code?**
    **A:** Accurate and efficient decomposition is numerically delicate. Production systems use optimized linear algebra libraries such as LAPACK, cuSOLVER, or framework wrappers.

## Explain it like a principal

Spectral thinking turns matrix behavior into directional behavior. Instead of asking whether a matrix is "large," ask which directions it amplifies, which it suppresses, how many directions matter, and how sensitive downstream computations are to perturbations. This is the bridge from linear algebra to model engineering: PCA, low-rank adapters, embedding compression, covariance diagnostics, Hessian approximations, and gradient stability all depend on spectra.

## Gotchas & follow-ups

- **Using eigendecomposition on any matrix without checking structure.** Non-symmetric matrices can have complex eigenvalues or defective eigenspaces. SVD is often the safer default.
- **Confusing eigenvalues and singular values.** Singular values are always nonnegative and apply to rectangular matrices; eigenvalues can be negative or complex and require square matrices.
- **Forgetting to center data for PCA.** Without centering, the first component may mostly capture the mean offset.
- **Assuming low reconstruction error means no product impact.** Rare directions can matter disproportionately for downstream tasks.
- **Follow-up prompt:** Derive why the right singular vectors of centered data are principal component directions.
