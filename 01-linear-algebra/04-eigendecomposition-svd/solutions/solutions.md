# Eigendecomposition and SVD — Solutions

---

## Solution 1 — Compute eigenpairs for a diagonal matrix

For

```
A = [[3, 0],
     [0, -2]]
```

the standard basis vectors are eigenvectors.

For e1 = `[1, 0]`:

```
A e1 = [3, 0] = 3 e1
```

So lambda1 = 3.

For e2 = `[0, 1]`:

```
A e2 = [0, -2] = -2 e2
```

So lambda2 = -2.

Geometrically, A stretches the x-axis direction by 3 and flips plus scales the y-axis direction by 2.

---

## Solution 2 — Verify an eigenpair

For v1 = `[1, 1]`:

```
B v1 = [[2, 1],
        [1, 2]] [1, 1]
     = [3, 3]
     = 3 [1, 1]
```

So v1 is an eigenvector with eigenvalue 3.

For v2 = `[1, -1]`:

```
B v2 = [[2, 1],
        [1, 2]] [1, -1]
     = [1, -1]
     = 1 [1, -1]
```

So v2 is an eigenvector with eigenvalue 1.

---

## Solution 3 — Reason about SVD shapes and rank

For A with shape `(100, 20)`, the reduced SVD with all 20 possible singular directions has:

```
U:       (100, 20)
Sigma:  (20,) or as a diagonal matrix (20, 20)
V^T:     (20, 20)
```

Because rank is 8, exactly 8 singular values are nonzero.

The best rank-5 approximation A_5 has the same shape as A:

```
(100, 20)
```

Rank 8 means the columns of A span an 8-dimensional subspace of R^100. Even though there are 20 columns, only 8 independent column directions exist.

---

## Solution 4 — Connect SVD to eigenvalues

Starting from:

```
A = U Sigma V^T
```

transpose A:

```
A^T = V Sigma^T U^T
```

Then:

```
A^T A = (V Sigma^T U^T)(U Sigma V^T)
      = V Sigma^T (U^T U) Sigma V^T
      = V Sigma^T Sigma V^T
```

Because U has orthonormal columns, `U^T U = I`. The diagonal entries of `Sigma^T Sigma` are squared singular values:

```
A^T A = V Sigma^2 V^T
```

Therefore, the right singular vectors of A are eigenvectors of `A^T A`, and the eigenvalues of `A^T A` are the squared singular values of A.

---

## Solution 5 — Interpret a spectrum

The matrix appears approximately low-rank because the first three singular values are much larger than the rest:

```
100, 72, 51, then 9, 3, 1, ...
```

There is a strong drop after rank 3, so rank 3 is a reasonable first compression candidate. Rank 4 may also be worth testing because the fourth singular value, 9, might still preserve useful signal.

Before deploying, evaluate downstream quality: retrieval relevance, classifier accuracy, calibration, or whatever product metric depends on the embeddings. Small singular values can encode rare but important information.

A very small smallest singular value suggests poor conditioning. Some input directions are nearly collapsed, so inverse or least-squares computations involving this matrix may be numerically unstable.
