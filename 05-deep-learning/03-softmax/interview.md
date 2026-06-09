# Softmax — Interview Prep

## Q&A

1. **Q: What does softmax do?**
   **A:** It maps logits to positive probabilities that sum to one, making them suitable for categorical distributions.

2. **Q: Why use logits instead of probabilities as model outputs?**
   **A:** Logits are unconstrained real numbers, easier for a network to produce and easier to optimize stably. Probabilities are derived when needed.

3. **Q: Why subtract the maximum logit before exponentiating?**
   **A:** It prevents overflow while preserving the output distribution because softmax is invariant to adding or subtracting a constant from all logits.

4. **Q: What is the softmax Jacobian?**
   **A:** `J_ij = p_i (1[i = j] - p_j)`. The diagonal terms are `p_i(1 - p_i)` and off-diagonal terms are `-p_i p_j`.

5. **Q: Why are softmax outputs coupled?**
   **A:** Increasing one logit changes the denominator, lowering other class probabilities even if their logits do not change.

6. **Q: What does temperature do?**
   **A:** Dividing logits by temperature changes sharpness. Lower temperature sharpens the distribution; higher temperature flattens it.

7. **Q: Why should you not call softmax before `CrossEntropyLoss` in PyTorch?**
   **A:** `CrossEntropyLoss` expects logits and internally applies a stable log-softmax plus negative log likelihood. Passing probabilities can degrade stability and produce wrong gradients.

8. **Q: Are softmax probabilities calibrated?**
   **A:** Not necessarily. They sum to one but may be overconfident or underconfident. Calibration must be evaluated separately.

9. **Q: How is softmax used in attention?**
   **A:** Attention scores are scaled and masked, then softmax normalizes them into weights over keys for each query.

10. **Q: What happens if all logits are equal?**
    **A:** Softmax returns the uniform distribution over classes.

## Explain it like a principal

Softmax is a normalization layer over alternatives, not a confidence guarantee. In production systems, the important decisions are whether the alternatives are truly mutually exclusive, whether logits are fed into the loss without premature normalization, how masks are represented before softmax, and whether downstream consumers treat probabilities as calibrated. For large vocabularies or long contexts, numerical stability and memory footprint also matter: you want fused log-softmax or attention kernels rather than materializing unstable intermediates.

## Gotchas & follow-ups

- **"Softmax picks the max class."** It produces a distribution; argmax is a separate operation.
- **Forgetting numerical stability.** Ask why subtracting `max(z)` does not change probabilities.
- **Confusing sigmoid and softmax.** Sigmoid is for independent binary labels; softmax is for mutually exclusive categories.
- **Assuming probability means calibration.** Softmax normalizes scores but does not prove correctness.
- **Applying masks incorrectly.** In attention, invalid positions should receive a very negative logit before softmax, not zero probability after an unstable computation.
