# Principal Component Analysis — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Center a dataset

Given rows `(1, 2)`, `(3, 4)`, and `(5, 6)`, compute the column means and centered matrix.

## Exercise 2 — Interpret eigenvalues

A covariance matrix has eigenvalues `[9, 3, 0]`. Compute the explained variance ratio for each component.

## Exercise 3 — Derive the first component objective

Show that maximizing variance of projected centered data `Xw` with `||w|| = 1` is equivalent to maximizing `w^T Sigma w`.

## Exercise 4 — Explain reconstruction

If data is projected onto the first `k` principal components and reconstructed, why does reconstruction error generally decrease as `k` increases?

## Exercise 5 — Prevent leakage

Why is it wrong to fit PCA on the full dataset before splitting train and validation data? What is the correct workflow?
