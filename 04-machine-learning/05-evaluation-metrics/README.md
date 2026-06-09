# Evaluation Metrics

## Overview

Evaluation metrics answer the question the loss function cannot answer alone: did the model solve the actual problem? Metrics define how model quality is judged on held-out data, production traffic, or business outcomes. A model can optimize its training `[[loss-functions]]` while failing the metric that matters.

Good AI engineers choose metrics from the decision context: class imbalance, asymmetric costs, ranking needs, calibration requirements, and acceptable failure modes.

## Math / Derivation

For binary classification:

| Metric | Formula | Measures |
|---|---|---|
| Accuracy | `(TP + TN) / N` | overall correctness |
| Precision | `TP / (TP + FP)` | correctness among predicted positives |
| Recall | `TP / (TP + FN)` | coverage of actual positives |
| F1 | `2PR / (P + R)` | harmonic mean of precision and recall |
| Specificity | `TN / (TN + FP)` | coverage of actual negatives |

Threshold-dependent metrics require choosing a decision threshold. Threshold-free summaries include ROC-AUC and PR-AUC.

For regression:

```
MAE  = (1/n) sum_i |yhat_i - y_i|
MSE  = (1/n) sum_i (yhat_i - y_i)^2
RMSE = sqrt(MSE)
R^2  = 1 - sum_i (y_i - yhat_i)^2 / sum_i (y_i - ybar)^2
```

For probabilistic classification, calibration metrics such as log loss, Brier score, and expected calibration error evaluate whether predicted probabilities match observed frequencies.

## Intuition

Accuracy asks, "how often are we right?" Precision asks, "when we raise an alarm, how often is it real?" Recall asks, "how many real cases did we catch?" F1 punishes systems that make one of precision or recall look good by sacrificing the other.

ROC-AUC asks whether positives tend to receive higher scores than negatives across thresholds. PR-AUC is usually more informative for rare positives because it focuses on positive predictions instead of the overwhelming number of true negatives.

## When & Why

Use accuracy only when classes are balanced and mistakes have similar cost. Use precision/recall tradeoffs when positive decisions trigger expensive or risky actions. Use PR-AUC for rare-event detection. Use calibration when probabilities drive downstream decisions, pricing, triage, or human review.

For regression, MAE is easier to interpret in target units and is robust to outliers. RMSE emphasizes large errors. R^2 compares against a mean baseline but can hide unacceptable tail behavior.

Always pair metrics with a split strategy that matches deployment: time-based splits for forecasting, group splits to avoid user leakage, and stratified splits for imbalanced classes.

## Implementation

A later implementation pass should implement confusion-matrix metrics, ROC-AUC, PR-AUC, MAE, MSE, RMSE, R^2, and simple calibration curves from tensors or arrays. It should validate calculations against trusted library outputs and include examples where accuracy is misleading under class imbalance.

The implementation should include threshold sweeps and show how changing a threshold moves precision and recall without retraining the model.

## Cross-links

- `[[loss-functions]]` — training objectives are proxies for evaluation goals.
- `[[logistic-regression]]` — classification thresholds turn probabilities into labels.
- `[[bias-variance]]` — validation metrics reveal underfitting and overfitting.
- `[[knn-naive-bayes]]` — simple classifiers expose metric tradeoffs clearly.
- `[[random-forests-xgboost]]` — tree ensembles are often evaluated with AUC, log loss, and calibration.

## Resources

- Tom Fawcett, "An introduction to ROC analysis." *Pattern Recognition Letters*, 2006.
- Jesse Davis and Mark Goadrich, "The Relationship Between Precision-Recall and ROC Curves." ICML 2006.
- scikit-learn User Guide, "Model evaluation: quantifying the quality of predictions."
