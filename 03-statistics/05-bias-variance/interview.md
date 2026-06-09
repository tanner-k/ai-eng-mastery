# Bias-Variance Tradeoff — Interview Prep

## Q&A

1. **Q: What is bias?**
   **A:** Bias is systematic error from the model class or learning procedure. A high-bias model cannot represent the true relationship well, even with many training samples.

2. **Q: What is variance?**
   **A:** Variance is sensitivity to the particular training data. A high-variance model changes substantially across samples and may overfit noise.

3. **Q: What is irreducible error?**
   **A:** It is noise in the target that no model can predict from the available features. In the squared-error decomposition, it is the observation noise variance.

4. **Q: How do train and validation error patterns diagnose bias and variance?**
   **A:** High train and validation error suggests high bias or optimization failure. Low train error with high validation error suggests high variance, overfitting, leakage, or distribution mismatch.

5. **Q: How does regularization affect the tradeoff?**
   **A:** Regularization constrains the model, often increasing bias slightly while reducing variance. The goal is lower validation error, not lower training error.

6. **Q: Why does more data usually reduce variance?**
   **A:** More data makes the learned model less dependent on idiosyncrasies of any one sample. It may not fix bias if the model class cannot represent the true relationship.

7. **Q: Why do random forests reduce variance?**
   **A:** They average many decorrelated decision trees. Individual trees are high variance, but averaging reduces prediction variance, especially when tree errors are not highly correlated.

8. **Q: What is double descent?**
   **A:** It is a modern generalization pattern where test error can decrease, then increase near the interpolation threshold, then decrease again as model capacity grows further. It complicates the classical U-shaped bias-variance story.

## Explain it like a principal

Bias-variance is a diagnostic framework, not a slogan. Principal-level use means mapping observed failures to interventions: add capacity or features for bias, add data or regularization for variance, inspect distribution shift when validation no longer predicts production, and separate irreducible noise from fixable modeling error. The train/validation gap is only the first clue; segment metrics, learning curves, seed variance, and calibration tell you which lever to pull.

## Gotchas & follow-ups

- **"Overfitting always means the model is too large."** It can also mean leakage, poor validation design, insufficient data, or weak regularization.
- **"More data fixes everything."** It mostly reduces variance. It does not fix missing features, wrong labels, or a model class with high bias.
- **"Regularization is always good."** Too much regularization increases bias and can underfit.
- **"Validation error is production error."** Only if validation data represents production traffic and labels.
- **Follow-up:** How would you decide whether to spend a month collecting data or improving model architecture?
