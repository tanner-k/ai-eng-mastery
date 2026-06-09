# Evaluation Metrics — Interview Prep

## Q&A

1. **Q: Why can accuracy be a bad metric?**
   **A:** It hides class imbalance and asymmetric costs. A majority-class classifier can have high accuracy while being useless.

2. **Q: Precision versus recall?**
   **A:** Precision measures how trustworthy positive predictions are. Recall measures how many actual positives are found.

3. **Q: When is PR-AUC better than ROC-AUC?**
   **A:** When positives are rare and the positive prediction quality matters most.

4. **Q: What does ROC-AUC measure?**
   **A:** The probability that a randomly chosen positive receives a higher score than a randomly chosen negative.

5. **Q: What is calibration?**
   **A:** Agreement between predicted probabilities and empirical frequencies. Among examples scored 0.8, roughly 80% should be positive.

6. **Q: Why tune thresholds on validation data?**
   **A:** The model score distribution and cost tradeoff determine the best threshold. The default 0.5 is often arbitrary.

7. **Q: MAE versus RMSE?**
   **A:** MAE weights each unit of error linearly. RMSE penalizes large errors more because it squares residuals before averaging.

8. **Q: What is data leakage in evaluation?**
   **A:** Information from validation or test examples influences training, feature engineering, or model selection, making metrics too optimistic.

9. **Q: Why use confidence intervals for metrics?**
   **A:** Metrics are estimates from finite samples. Confidence intervals show uncertainty and prevent overreacting to noise.

## Explain it like a principal

Evaluation design is part of system design. The metric must reflect the decision being automated, the cost of mistakes, the deployment distribution, and the population segments where failures matter. Principal-level metric work includes thresholding policy, calibration, split strategy, subgroup analysis, confidence intervals, and monitoring for production drift. A single leaderboard number is rarely enough.

## Gotchas & follow-ups

- **"AUC is threshold independent, so thresholding does not matter."** Deployment still needs a threshold or ranking cutoff.
- **"F1 is always the best imbalance metric."** F1 ignores true negatives and assumes precision and recall are equally important.
- **"High log loss means low accuracy."** Not necessarily. Log loss also punishes poor probability calibration and confident mistakes.
- **Follow-up:** How would you evaluate a model whose top 100 predictions are reviewed by humans each day?
