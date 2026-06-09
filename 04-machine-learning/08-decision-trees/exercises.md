# Decision Trees — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Compute Gini impurity

Compute Gini impurity for a node with class counts `[6, 4]` and for a pure node with counts `[10, 0]`.

## Exercise 2 — Compute split gain

A parent node has 10 examples with counts `[6, 4]`. A split creates a left child with counts `[5, 1]` and a right child with counts `[1, 3]`. Compute the weighted Gini gain.

## Exercise 3 — Regression leaf prediction

A regression leaf contains targets `[2, 4, 9]`. What prediction minimizes squared error in the leaf, and what is the SSE?

## Exercise 4 — Diagnose overfitting

A tree has near-zero training error but poor validation error. Name three hyperparameters or methods that can reduce overfitting.

## Exercise 5 — Explain interaction modeling

Give a short example of how a decision tree can represent a feature interaction that a plain linear model would miss without engineered interaction terms.
