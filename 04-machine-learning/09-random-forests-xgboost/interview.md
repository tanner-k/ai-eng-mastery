# Random Forests and XGBoost — Interview Prep

## Q&A

1. **Q: What is bagging?**
   **A:** Bootstrap aggregating: train models on bootstrap samples and average or vote their predictions.

2. **Q: Why does a random forest use feature subsampling?**
   **A:** To decorrelate trees. Averaging helps most when individual models make different errors.

3. **Q: Random forest versus boosting?**
   **A:** Forests train trees independently and average them. Boosting trains trees sequentially to correct current errors.

4. **Q: What are pseudo-residuals in gradient boosting?**
   **A:** Negative gradients of the loss with respect to current predictions.

5. **Q: What does learning rate do in boosting?**
   **A:** It shrinks each tree's contribution, requiring more trees but often improving generalization.

6. **Q: Why is early stopping useful?**
   **A:** It stops adding trees when validation performance no longer improves, controlling overfit.

7. **Q: Why is XGBoost strong on tabular data?**
   **A:** It combines boosted trees with regularization, efficient split finding, missing-value handling, and strong systems optimizations.

8. **Q: What are common boosted-tree hyperparameters?**
   **A:** Number of trees, learning rate, max depth, min child weight, subsample, column sample, and regularization terms.

9. **Q: Are feature importances reliable?**
   **A:** They are useful diagnostics but can be biased by cardinality, correlation, and leakage.

## Explain it like a principal

Tree ensembles are usually the first serious answer for tabular prediction. Random forests buy robustness by averaging unstable trees. Boosting buys accuracy by sequentially fitting residual structure with regularization and early stopping. Production maturity means guarding against leakage, using time-aware validation when needed, monitoring calibration, and treating feature importance as a diagnostic rather than proof.

## Gotchas & follow-ups

- **"More trees always overfit a random forest."** More forest trees usually stabilize the average; overfit is more controlled by tree depth and leaf size.
- **"Boosting just averages trees."** Boosting is sequential and loss-gradient driven.
- **"XGBoost handles missing values so preprocessing does not matter."** Missingness can be signal or leakage; validation still matters.
- **Follow-up:** Why might a boosted model beat a neural network on a medium-sized tabular dataset?
