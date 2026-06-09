# MLE and Negative Log-Likelihood — Interview Prep

## Q&A

1. **Q: What is maximum likelihood estimation?**
   **A:** MLE chooses model parameters that maximize the probability of the observed data under the assumed model family.

2. **Q: Why minimize negative log-likelihood instead of maximize likelihood?**
   **A:** Logs turn products into sums, improving numerical stability and making per-example losses additive. The negative sign converts maximization into the minimization convention used by optimizers.

3. **Q: How does Bernoulli NLL relate to binary cross-entropy?**
   **A:** They are the same objective: `-[y log p + (1-y) log(1-p)]` summed or averaged over examples.

4. **Q: How does Gaussian NLL relate to MSE?**
   **A:** With fixed variance, Gaussian NLL is a constant plus a positive multiple of squared residuals. Therefore the minimizer is the same as MSE.

5. **Q: What does NLL penalize in classification?**
   **A:** It penalizes the negative log probability assigned to the true class. Confident wrong predictions receive very large penalties.

6. **Q: What is model misspecification?**
   **A:** The assumed likelihood family does not match the real data-generating process. MLE then finds the best parameters within the wrong family, which can still be systematically bad.

7. **Q: Why are logits preferred over probabilities for cross-entropy implementations?**
   **A:** Logit-based implementations combine sigmoid or softmax with log operations using stable algebra. Direct probabilities can hit `0` or `1`, causing `log(0)` or poor gradients.

8. **Q: Is MLE Bayesian?**
   **A:** No. MLE uses the likelihood only. Bayesian inference combines likelihood with a prior to form a posterior; MAP estimation adds a prior term but still returns a point estimate.

## Explain it like a principal

NLL is the probabilistic contract behind many ML losses. At principal level, the critical question is whether the likelihood matches the product behavior you need. Cross-entropy is right when labels are categorical and calibrated probabilities matter; Gaussian NLL is right when residuals are roughly symmetric with stable variance. When the assumption is wrong, optimization can look healthy while the model learns the wrong notion of error.

## Gotchas & follow-ups

- **"Low NLL always means high accuracy."** Not necessarily. NLL also rewards calibration and confidence on correct examples.
- **"MSE is just an arbitrary regression loss."** It corresponds to fixed-variance Gaussian likelihood.
- **"Taking logs changes the optimum."** Log is monotonic, so maximizing likelihood and log-likelihood have the same optimum.
- **"Class imbalance is solved by likelihood."** Plain likelihood reflects the observed distribution; cost-sensitive behavior may need weighting, resampling, or a different decision threshold.
- **Follow-up:** How would you modify a likelihood-based objective for heteroskedastic regression?
