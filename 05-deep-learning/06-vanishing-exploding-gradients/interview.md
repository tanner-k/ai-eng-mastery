# Vanishing and Exploding Gradients — Interview Prep

## Q&A

1. **Q: What are vanishing gradients?**
   **A:** Gradients shrink as they propagate backward, leaving early layers with near-zero updates.

2. **Q: What are exploding gradients?**
   **A:** Gradients grow rapidly through the backward pass, causing unstable updates, overflow, or `nan` values.

3. **Q: What mathematical structure causes both?**
   **A:** Products of Jacobians across depth or time. Repeated factors below one vanish; repeated factors above one explode.

4. **Q: Why are recurrent networks especially vulnerable?**
   **A:** They reuse the same transition across many time steps, so the same Jacobian is multiplied repeatedly.

5. **Q: How does tanh contribute to vanishing gradients?**
   **A:** In saturation, `tanh'(x) = 1 - tanh(x)^2` is near zero, so the backward signal is dampened.

6. **Q: Why does ReLU help but not solve everything?**
   **A:** Active ReLU units have derivative one, preserving gradient, but inactive units have derivative zero and can die.

7. **Q: What is gradient clipping?**
   **A:** It rescales gradients when their norm exceeds a threshold, limiting update magnitude while preserving direction for global norm clipping.

8. **Q: How do residual connections help?**
   **A:** They provide identity paths so gradients can flow through additions instead of only through deep nonlinear transformations.

9. **Q: What telemetry would you monitor?**
   **A:** Gradient norms by layer, activation distributions, update-to-weight ratios, loss spikes, and nonfinite values.

10. **Q: Can adaptive optimizers fix vanishing gradients?**
    **A:** Not fully. They can rescale available gradients, but they cannot recover information if upstream gradients are effectively zero.

## Explain it like a principal

Vanishing and exploding gradients are signal-propagation failures. The right response is not just "lower the learning rate" or "use clipping"; it is to locate where the signal changes scale and choose an intervention at that level. Initialization and normalization address variance propagation, residual paths address depth, activation choice addresses local derivatives, and clipping addresses unstable update magnitude. Principal-level debugging connects gradient telemetry to architecture changes rather than treating gradient norms as isolated numbers.

## Gotchas & follow-ups

- **"Exploding gradients are only a high learning-rate problem."** Learning rate affects updates, but gradient explosion can originate in the backward graph before the optimizer step.
- **"Adam fixes small gradients."** Adam rescales but cannot create a meaningful signal through saturated layers.
- **Confusing activation explosion with gradient explosion.** They often co-occur but should be measured separately.
- **Using clipping as the only solution.** Clipping masks explosions; it may not fix the architecture causing them.
- **Ignoring layer-wise diagnostics.** A global norm can hide which layer is failing.

Follow-up: How would you distinguish a bad learning rate from a depth-related gradient propagation problem using only training telemetry?
