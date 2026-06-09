# Random Forests and XGBoost — Solutions

## Solution 1 — Explain bagging

Deep trees have high variance: small data changes can alter their structure. Bootstrap samples create different trees, and averaging reduces variance. If tree errors were independent, variance would fall roughly by `1/B`. In practice errors are correlated, so feature subsampling helps decorrelate trees and makes averaging more effective.

## Solution 2 — Compute an averaged prediction

```
(10 + 12 + 9 + 11 + 13) / 5 = 55 / 5 = 11
```

The forest prediction is `11`.

## Solution 3 — Derive squared-error boosting residuals

For `L = (1/2)(y - F)^2`:

```
dL/dF = -(y - F) = F - y
```

The negative gradient is:

```
-dL/dF = y - F
```

which is the residual.

## Solution 4 — Diagnose boosting overfit

Use early stopping, reduce learning rate, reduce max depth, increase minimum child/leaf size, add row or column subsampling, increase regularization, or reduce the number of trees selected by validation.

## Solution 5 — Choose forest or boosting

Start with a random forest. It is robust, parallelizable, and less sensitive to careful hyperparameter tuning. Boosted trees may later achieve better accuracy, but they require more validation discipline around learning rate, depth, number of trees, and leakage.
