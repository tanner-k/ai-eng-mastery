# Evaluation Metrics — Solutions

## Solution 1 — Compute confusion-matrix metrics

Total `N = 1000`.

```
accuracy = (40 + 930) / 1000 = 0.97
precision = 40 / (40 + 10) = 0.80
recall = 40 / (40 + 20) = 0.6667
specificity = 930 / (930 + 10) = 0.9894
F1 = 2 * 0.8 * 0.6667 / (0.8 + 0.6667) = 0.7273
```

## Solution 2 — Explain accuracy failure

Accuracy is 99% because 99% of examples are negative and all are predicted negative. Precision is undefined because there are no predicted positives; many systems report it as 0 with `zero_division=0`. Recall is 0 because no actual positives are found. Accuracy is misleading because the model completely fails the rare class.

## Solution 3 — Compare MAE and RMSE

Model A:

```
MAE = 1
RMSE = sqrt((1+1+1+1)/4) = 1
```

Model B:

```
MAE = (0+0+0+4)/4 = 1
RMSE = sqrt(16/4) = 2
```

RMSE exposes the tail error more strongly.

## Solution 4 — Choose a metric

Monitor recall/sensitivity, false-negative rate, PR-AUC, and precision at clinically acceptable recall. Threshold selection should target high recall subject to a tolerable false-positive or review burden. Calibration is also important if scores guide risk communication or triage.

## Solution 5 — Interpret threshold movement

Lowering the threshold usually increases recall because more examples are labeled positive, so fewer actual positives are missed. Precision often decreases because more negatives also cross the threshold, increasing false positives. The exact movement depends on score distributions, but this is the usual tradeoff.
