# Probability Basics — Interview Prep

## Q&A

1. **Q: What is the difference between probability and odds?**
   **A:** Probability is `p`, the chance an event occurs. Odds are `p / (1 - p)`, the chance it occurs relative to the chance it does not. Logistic regression models log-odds because they are unconstrained on the real line.

2. **Q: What does conditional probability mean?**
   **A:** It is the probability of an event after restricting attention to cases where another event occurred: `P(A | B) = P(A and B) / P(B)`.

3. **Q: What does independence mean?**
   **A:** Events are independent if observing one does not change the probability of the other, equivalently `P(A and B) = P(A)P(B)` when both probabilities are defined.

4. **Q: Why is Bayes' rule useful in ML?**
   **A:** It converts evidence likelihoods and priors into posterior beliefs. It is the basis for Bayesian inference, diagnostic reasoning, and understanding false positives under class imbalance.

5. **Q: Why can high detector accuracy be misleading for rare events?**
   **A:** When the base rate is low, false positives can dominate positive predictions even with high sensitivity. Precision depends on prevalence, not just true-positive rate.

6. **Q: What is expectation?**
   **A:** It is the probability-weighted average value of a random variable. For model metrics, it represents the average result over repeated samples from the same distribution.

7. **Q: How does variance differ from standard deviation?**
   **A:** Variance is expected squared deviation from the mean. Standard deviation is its square root and is in the same units as the original variable.

8. **Q: Why should validation metrics be treated as random variables?**
   **A:** They depend on the sampled validation set. Another sample from the same population could produce a different value, so uncertainty intervals matter.

## Explain it like a principal

Probability is the discipline that keeps ML decisions honest under uncertainty. A principal engineer should use it to separate observed metrics from underlying population behavior, and to avoid base-rate mistakes in production systems. The most important habit is conditioning on the right event: "accuracy given the validation set" is not the same as "future performance under shifted traffic," and "detector fired" is not the same as "defect is present."

## Gotchas & follow-ups

- **"95% accurate means 95% of positives are real."** False under class imbalance. Positive predictive value depends on the base rate.
- **"Independent means mutually exclusive."** False. Mutually exclusive nonempty events are negatively dependent.
- **"The validation metric is the true metric."** It is an estimate with sampling error.
- **"Correlation implies dependence but explains causation."** Correlation can indicate dependence, but causal claims require stronger assumptions.
- **Follow-up:** How would you explain a high false-alarm rate for a rare-event classifier to a product team?
