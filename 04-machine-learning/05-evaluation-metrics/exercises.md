# Evaluation Metrics — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Compute confusion-matrix metrics

Given `TP = 40`, `FP = 10`, `FN = 20`, and `TN = 930`, compute accuracy, precision, recall, specificity, and F1.

## Exercise 2 — Explain accuracy failure

A dataset has 1% positives. A model predicts every example as negative. What is its accuracy, precision, and recall? Why is accuracy misleading?

## Exercise 3 — Compare MAE and RMSE

Two regression models have residuals `A = [1, 1, 1, 1]` and `B = [0, 0, 0, 4]`. Compute MAE and RMSE for both. Which metric exposes the tail error more strongly?

## Exercise 4 — Choose a metric

For a medical screening model where missed disease cases are much worse than false alarms, which metrics would you monitor and optimize threshold selection against?

## Exercise 5 — Interpret threshold movement

A classifier threshold is lowered from `0.8` to `0.3`. In general, what happens to precision and recall? Explain why.
