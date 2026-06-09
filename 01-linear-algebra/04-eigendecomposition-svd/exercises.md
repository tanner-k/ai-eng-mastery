# Eigendecomposition and SVD — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Compute eigenpairs for a diagonal matrix

Let

```
A = [[3, 0],
     [0, -2]]
```

Find two eigenvalues and corresponding eigenvectors. Explain geometrically what A does to each eigenvector direction.

---

## Exercise 2 — Verify an eigenpair

Let

```
B = [[2, 1],
     [1, 2]]
```

Show that v1 = `[1, 1]` and v2 = `[1, -1]` are eigenvectors of B. Find their eigenvalues.

---

## Exercise 3 — Reason about SVD shapes and rank

A matrix A has shape `(100, 20)` and rank 8.

1. In the reduced SVD `A = U Sigma V^T`, what are the shapes of U, Sigma, and V^T if all 20 possible singular directions are returned?
2. How many singular values are nonzero?
3. What is the shape of the best rank-5 approximation A_5?
4. What does rank 8 imply about the column space of A?

---

## Exercise 4 — Connect SVD to eigenvalues

Let A be any real matrix. Starting from `A = U Sigma V^T`, derive the eigendecomposition of `A^T A` in terms of V and Sigma. What does this imply about the relationship between singular values of A and eigenvalues of `A^T A`?

---

## Exercise 5 — Interpret a spectrum

An embedding matrix has singular values:

```
100, 72, 51, 9, 3, 1, 0.2, 0.05
```

1. Is the matrix approximately low-rank? Why?
2. What rank might you try first for compression?
3. What risk should you evaluate before deploying a low-rank approximation?
4. What does a very small smallest singular value suggest about conditioning?
