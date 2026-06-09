# Cross-Entropy and NLL Loss — Interview Prep

## Q&A

1. **Q: What is cross-entropy loss for classification?**
   **A:** It is the negative log probability assigned to the correct class, usually computed from logits through a stable log-softmax.

2. **Q: How is NLL related to cross-entropy?**
   **A:** With one-hot labels, cross-entropy equals the negative log likelihood of the correct class. NLL loss expects log probabilities; cross-entropy usually expects logits.

3. **Q: Why should logits be passed directly to cross-entropy?**
   **A:** The fused operation computes log-softmax stably and avoids overflow, underflow, and altered gradients from manually applying softmax.

4. **Q: What is the gradient with respect to logits?**
   **A:** For one example with one-hot target, it is `p - y`, where `p = softmax(z)`.

5. **Q: Why does cross-entropy penalize confident wrong predictions strongly?**
   **A:** Because `-log p_c` grows rapidly as the correct-class probability approaches zero.

6. **Q: What is `logsumexp` and why does it matter?**
   **A:** It is `log(sum_j exp(z_j))` computed stably, usually by subtracting the maximum logit first. It is the normalization term in log-softmax.

7. **Q: How do class weights affect cross-entropy?**
   **A:** They multiply selected examples or classes, changing both loss contribution and gradient magnitude for those labels.

8. **Q: When is binary cross-entropy better than softmax cross-entropy?**
   **A:** When labels are independent and multiple classes can be true at once, such as multi-label tagging.

9. **Q: What does `ignore_index` do?**
   **A:** It excludes selected targets from loss and gradient computation, commonly for padded sequence tokens.

10. **Q: Why can accuracy improve while cross-entropy worsens?**
    **A:** Accuracy only checks argmax. Cross-entropy also measures probability assigned to the correct class and can worsen if predictions become overconfident on mistakes.

## Explain it like a principal

Cross-entropy is a likelihood objective with sharp operational consequences. It defines what probability mass the model is rewarded for assigning, how examples are weighted, and how gradient scale changes with reduction. In production training, many issues come from interface mismatches: probabilities passed where logits are expected, padding tokens contributing to loss, class weights changing effective learning rate, or metrics masking calibration problems. A principal-level answer should connect the formula to the data contract and the training telemetry.

## Gotchas & follow-ups

- **Passing softmax probabilities into a logits loss.** This is one of the most common classifier bugs.
- **Confusing NLL input type.** NLL loss expects log probabilities, not raw probabilities.
- **Ignoring reduction mode.** `sum` scales gradients with batch size; `mean` normalizes them.
- **Using softmax for multi-label tasks.** Softmax forces competition between labels that may co-occur.
- **Equating low loss with high accuracy.** They correlate but optimize different signals.

Follow-up: Given a batch with padding, class imbalance, and mixed reduction settings, how would you verify the loss scale and gradient contribution of each example?
