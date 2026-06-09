# Logistic Regression — Interview Prep

## Q&A

1. **Q: Why is it called regression if it is a classifier?**
   **A:** It regresses log-odds as a linear function of features, then converts them to probabilities.

2. **Q: What loss does logistic regression use?**
   **A:** Binary cross-entropy, equivalent to Bernoulli negative log-likelihood.

3. **Q: What is a logit?**
   **A:** The unbounded score `log(p/(1-p))` before applying sigmoid.

4. **Q: What is the decision boundary?**
   **A:** For threshold 0.5, it is the hyperplane `x^T w + b = 0`.

5. **Q: Why not use MSE for classification?**
   **A:** BCE matches Bernoulli likelihood and gives better gradients for probability classification.

6. **Q: How do you interpret coefficients?**
   **A:** A coefficient is the change in log-odds for a one-unit feature increase, holding other features fixed.

7. **Q: How do you handle class imbalance?**
   **A:** Use class weights, resampling, threshold tuning, and metrics such as PR-AUC, precision, and recall.

8. **Q: Is logistic regression calibrated?**
   **A:** It often calibrates well when the model is correctly specified and regularized, but calibration should still be measured.

9. **Q: What happens with perfect separation?**
   **A:** The likelihood keeps improving as coefficient magnitudes grow, so regularization or early stopping is needed.

## Explain it like a principal

Logistic regression is the simplest serious probability model for binary decisions. It gives a score, a probability, and an interpretable linear boundary. In production, it is valuable because failures are inspectable: feature weights, calibration curves, threshold tradeoffs, and regularization effects are all visible. A senior answer should connect likelihood, log-odds, thresholding, imbalance, and coefficient stability.

## Gotchas & follow-ups

- **"Threshold 0.5 is the natural threshold."** It is only natural under equal costs and calibrated probabilities.
- **"A coefficient is a probability change."** It is a log-odds change; probability effects depend on the current baseline probability.
- **"High accuracy means good classifier."** Rare positives can make accuracy meaningless.
- **Follow-up:** How would you deploy one logistic model for scoring but different thresholds for different review queues?
