# Label Smoothing — Interview Prep

## Q&A

1. **Q: What is label smoothing?**
   **A:** It replaces one-hot targets with softened target distributions that keep most mass on the correct class and assign small mass to other classes.

2. **Q: Why can label smoothing improve generalization?**
   **A:** It discourages extreme confidence and large logit gaps, reducing overfitting to hard or noisy labels.

3. **Q: What is the common formula?**
   **A:** `q = (1 - epsilon) * one_hot + epsilon / K`, though some conventions distribute epsilon only over incorrect classes.

4. **Q: How does it change the cross-entropy gradient?**
   **A:** The gradient becomes `p - q` instead of `p - y`, where `q` is the smoothed target.

5. **Q: Does label smoothing change the architecture?**
   **A:** No. It changes the training target and loss signal, not the model's forward computation.

6. **Q: Can label smoothing hurt calibration?**
   **A:** It can reduce overconfidence, but it does not guarantee calibration and can distort probability estimates in some settings.

7. **Q: Why can it conflict with knowledge distillation?**
   **A:** Teacher probabilities already encode soft relationships among classes. Uniform smoothing can erase or dilute that information.

8. **Q: How does smoothing affect the optimal logits?**
   **A:** It prevents the optimum from requiring infinite logit gaps because the target assigns nonzero probability to other classes.

9. **Q: Is label smoothing the same as adding label noise?**
   **A:** No. It deterministically changes targets; it does not randomly corrupt labels, although both can regularize.

10. **Q: What metrics would you check when tuning epsilon?**
    **A:** Accuracy, validation NLL, expected calibration error, confidence histograms, and class-specific performance.

## Explain it like a principal

Label smoothing is a target-design decision. It encodes the belief that hard labels should not force infinite certainty, which often improves robustness but changes the meaning of the optimized likelihood. Principal-level use requires checking the data source, downstream probability requirements, class imbalance, and whether targets already contain information richer than one-hot labels. The right smoothing value is empirical, but the risk analysis is conceptual: do you want less overconfidence, or do you need the model's probabilities to match observed frequencies as closely as possible?

## Gotchas & follow-ups

- **Convention mismatch.** Some libraries spread epsilon across all classes; others spread only across incorrect classes.
- **"It always improves calibration."** It often reduces confidence, but calibration must be measured.
- **Using it blindly with distillation.** Teacher distributions are already soft targets.
- **Ignoring rare classes.** Smoothing can weaken scarce positive signal.
- **Confusing target smoothing with output temperature.** Smoothing changes training targets; temperature rescales logits.

Follow-up: How would you decide whether a drop in validation NLL but worse calibration after smoothing is acceptable for a production classifier?
