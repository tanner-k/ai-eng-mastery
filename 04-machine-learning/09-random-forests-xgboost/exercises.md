# Random Forests and XGBoost — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Explain bagging

Why does averaging many bootstrap-trained trees reduce variance? Include the role of tree correlation.

## Exercise 2 — Compute an averaged prediction

Five regression trees predict `[10, 12, 9, 11, 13]` for the same example. What is the random forest prediction?

## Exercise 3 — Derive squared-error boosting residuals

For loss `L = (1/2)(y - F(x))^2`, show that the negative gradient with respect to `F(x)` is the residual `y - F(x)`.

## Exercise 4 — Diagnose boosting overfit

Training loss keeps falling while validation loss starts rising after 200 trees. What controls would you tune?

## Exercise 5 — Choose forest or boosting

You need a fast, robust baseline on a noisy tabular dataset with limited tuning time. Would you start with a random forest or boosted trees? Explain.
