# Loss Functions — Interview Prep

## Q&A

1. **Q: What is a loss function?**
   **A:** It is a scalar objective that measures model error on training examples. Optimization algorithms use its gradient to update parameters, so the loss defines the training geometry.

2. **Q: How is a loss different from an evaluation metric?**
   **A:** A loss must usually be differentiable and useful for optimization. A metric measures business or modeling success and may be non-differentiable, such as F1, accuracy, recall at k, or AUC.

3. **Q: Why is MSE sensitive to outliers?**
   **A:** The penalty grows quadratically and the gradient grows linearly with residual size. A few extreme residuals can dominate the update.

4. **Q: Why is cross-entropy preferred for classification?**
   **A:** It is the negative log-likelihood of the true class under the predicted distribution. It strongly penalizes confident wrong predictions and gives useful gradients when probabilities are wrong.

5. **Q: What is the benefit of using logits-based cross-entropy implementations?**
   **A:** They avoid unstable operations such as taking `log(sigmoid(z))` directly. Stable implementations combine sigmoid or softmax with log operations using algebraically equivalent forms.

6. **Q: When would you use Huber loss?**
   **A:** When regression residuals are mostly small and smooth optimization is useful, but occasional outliers should not dominate as much as they do under MSE.

7. **Q: What probabilistic assumption underlies MSE?**
   **A:** Independent Gaussian residuals with constant variance. Minimizing MSE is equivalent to maximum likelihood under that assumption.

8. **Q: Why can optimizing accuracy directly be difficult?**
   **A:** Accuracy depends on discrete thresholded predictions, so it is flat almost everywhere with respect to model parameters. Cross-entropy provides a smooth surrogate.

## Explain it like a principal

Loss choice is a modeling decision, not just a training detail. It encodes what errors cost, what distributional assumption you are making, and what gradient signal the optimizer receives. At production scale, the wrong loss can create a model that looks healthy in training telemetry but fails the actual product requirement: for example, optimizing MSE when tail errors matter, or optimizing unweighted cross-entropy when rare positives drive business value. A strong answer connects loss, metric, data distribution, calibration, and operational cost.

## Gotchas & follow-ups

- **"Use the metric as the loss."** Sometimes impossible or unwise. Many metrics are non-differentiable or too noisy, so use a surrogate and validate the real metric.
- **"MSE is always for regression."** MSE trains conditional means; it can be poor for heavy-tailed targets or asymmetric costs.
- **"Cross-entropy guarantees calibrated probabilities."** It encourages probabilistic predictions, but calibration still depends on data, model capacity, regularization, and post-processing.
- **Follow-up:** How would you design a loss when false negatives are ten times more expensive than false positives?
