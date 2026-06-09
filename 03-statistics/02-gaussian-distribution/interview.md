# Gaussian Distribution — Interview Prep

## Q&A

1. **Q: Why is the Gaussian distribution so common?**
   **A:** Averages and sums of many small independent effects tend toward Gaussian behavior under central-limit conditions. The Gaussian is also mathematically convenient: it is fully described by mean and covariance and leads to tractable likelihoods.

2. **Q: What do `mu` and `sigma^2` represent?**
   **A:** `mu` is the mean and center of the distribution. `sigma^2` is the variance, measuring average squared spread around the mean; `sigma` is the standard deviation.

3. **Q: What is standardization?**
   **A:** It maps `X ~ N(mu, sigma^2)` to `Z = (X - mu) / sigma`, which follows `N(0, 1)`. It expresses values in standard deviation units.

4. **Q: Why does Gaussian noise imply squared-error loss?**
   **A:** The Gaussian negative log-likelihood contains a constant term plus `(y - yhat)^2 / (2 sigma^2)`. With fixed variance, minimizing NLL is the same as minimizing squared error.

5. **Q: What is covariance in a multivariate Gaussian?**
   **A:** Covariance describes feature variances and pairwise linear relationships. It determines the orientation and shape of the probability ellipsoid.

6. **Q: What is Mahalanobis distance?**
   **A:** It is `(x - mu)^T Sigma^{-1}(x - mu)`, a covariance-aware squared distance. It downweights directions with high variance and accounts for correlated features.

7. **Q: When is a Gaussian assumption poor?**
   **A:** It is poor for strongly skewed, bounded, count-valued, multimodal, or heavy-tailed data. It can understate tail risk and over-penalize valid outliers.

8. **Q: Why does outlier sensitivity matter?**
   **A:** Gaussian NLL grows quadratically with residual size. A few extreme points can dominate fitting and distort model evaluation if the actual noise is heavy-tailed.

## Explain it like a principal

The Gaussian is a useful default, not a law. Principal-level judgment is knowing when its convenience aligns with the data-generating process and when it hides tail risk. In production, the practical questions are whether residuals are symmetric, whether variance is stable across segments, and whether anomaly thresholds are calibrated to real tail behavior. Gaussian assumptions are often good enough for first-pass modeling, but they should be checked with residual plots, calibration, and segment-level diagnostics.

## Gotchas & follow-ups

- **"Gaussian means most values are close to the mean."** True qualitatively, but tail probabilities still matter at production scale.
- **"Zero correlation implies independence."** Only generally true under joint Gaussian assumptions, not for arbitrary distributions.
- **"MSE is always justified by Gaussian noise."** MSE corresponds to fixed-variance Gaussian noise; heteroskedastic or heavy-tailed noise weakens that justification.
- **"Diagonal covariance is harmless."** It ignores feature correlations and can mis-score anomalies along correlated directions.
- **Follow-up:** How would you test whether residuals from a regression model are plausibly Gaussian?
