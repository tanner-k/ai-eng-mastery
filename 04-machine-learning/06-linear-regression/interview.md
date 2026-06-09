# Linear Regression — Interview Prep

## Q&A

1. **Q: What is linear regression optimizing?**
   **A:** Usually the sum or mean of squared residuals between predictions and targets.

2. **Q: What are the normal equations?**
   **A:** `X^T X beta = X^T y`, obtained by setting the least-squares gradient to zero.

3. **Q: Why avoid explicitly computing `(X^T X)^{-1}`?**
   **A:** Matrix inversion is less stable and often slower than solving the linear system directly.

4. **Q: What does a coefficient mean?**
   **A:** It is the expected target change for a one-unit feature increase holding other included features fixed, subject to modeling assumptions.

5. **Q: What is multicollinearity?**
   **A:** Strong linear dependence among features. It makes coefficient estimates unstable even if predictions remain good.

6. **Q: What does ridge regression fix?**
   **A:** It shrinks coefficients and stabilizes estimates when features are correlated or data is limited.

7. **Q: What residual assumptions matter?**
   **A:** Mean-zero errors, constant variance, independence, and a roughly appropriate linear functional form.

8. **Q: Is linear regression only useful for linear data?**
   **A:** No. Feature engineering can make nonlinear relationships linear in transformed features.

9. **Q: How do you evaluate regression?**
   **A:** Use held-out MAE/RMSE/R^2 and inspect residuals, tail errors, and subgroup behavior.

## Explain it like a principal

Linear regression is the control model for supervised learning. It gives fast baselines, interpretable coefficients, closed-form diagnostics, and a clean way to reason about loss geometry. In production, its value often comes from being boring: you can explain it, monitor it, regularize it, and compare complex models against it. The mature answer covers numerical stability, feature scale, collinearity, residual behavior, and whether coefficient interpretation is causal or merely associational.

## Gotchas & follow-ups

- **"Closed form means exact and always best."** It can be numerically unstable or infeasible at high dimension; regularization and stable solvers matter.
- **"Low RMSE means the model is good."** Tail errors, leakage, and subgroup failures can hide under aggregate metrics.
- **"Coefficients are causal effects."** Not without assumptions about confounding and data generation.
- **Follow-up:** What changes when `d > n`, and why does ridge still work?
