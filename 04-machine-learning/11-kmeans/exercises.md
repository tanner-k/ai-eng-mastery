# K-Means — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Assign points to centroids

In one dimension, points are `[0, 2, 9, 10]` and centroids are `mu_1 = 1`, `mu_2 = 8`. Assign each point to its nearest centroid.

## Exercise 2 — Update centroids

Using the assignments from Exercise 1, compute the updated centroids.

## Exercise 3 — Prove the mean update

Show that the mean of assigned points minimizes `sum_i ||x_i - mu||_2^2` for a fixed cluster.

## Exercise 4 — Diagnose scale sensitivity

Why can one feature measured in dollars dominate another feature measured as a rate between 0 and 1? What preprocessing helps?

## Exercise 5 — Choose K

An elbow plot shows inertia decreasing sharply until `K=4` and slowly afterward. What does this suggest, and why is it not definitive proof that four real clusters exist?
