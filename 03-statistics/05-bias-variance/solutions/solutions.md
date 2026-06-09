# Bias-Variance Tradeoff — Solutions

## Solution 1 — Identify bias and variance regimes

1. High training and validation error suggests high bias, optimization failure, or severe feature problems.
2. Near-zero training error with much higher validation error suggests high variance or overfitting.
3. Low training and validation error suggests the current model is working well under the validation distribution.
4. Moderate training error, high validation error, and large seed-to-seed variation suggests high variance, possibly with some bias if training error remains meaningfully high.

## Solution 2 — Compute a simple decomposition

Squared bias:

```text
(E[f_hat(x)] - f(x))^2 = (8 - 10)^2 = 4
```

Expected squared prediction error:

```text
bias^2 + variance + noise
= 4 + 3 + 2
= 9
```

## Solution 3 — Reason about model capacity

Degree `1` is most likely high bias because a line cannot represent a cubic relationship. Degree `20` is most likely high variance because it can fit noise and oscillate between points. Degree `3` is likely best because it matches the true smooth cubic form, assuming enough data to estimate it reliably.

## Solution 4 — Explain ensembling

Averaging reduces variance because independent prediction errors cancel out. If each model has variance `sigma^2` and errors are independent, the average of `M` models has variance `sigma^2 / M`.

The reduction is strongest when model errors are uncorrelated or weakly correlated. If all models make the same errors, averaging provides little variance reduction.

## Solution 5 — Diagnose a production regression

It is not necessarily a classical bias-variance problem. Possible causes include distribution shift in the new traffic segment, label-definition mismatch, missing segment-specific features, training-serving skew, or validation leakage that made offline results too optimistic.

A useful diagnostic is segment-level evaluation: compare feature distributions, label distributions, calibration, and error types for the new segment versus the validation set. If validation did not represent the segment, the issue is primarily evaluation and distribution coverage rather than just capacity or variance.
