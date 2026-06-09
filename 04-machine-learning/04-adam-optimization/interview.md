# Adam Optimization — Interview Prep

## Q&A

1. **Q: What does Adam track?**
   **A:** An exponential moving average of gradients and an exponential moving average of squared gradients.

2. **Q: Why does Adam need bias correction?**
   **A:** The moving averages start at zero, so early estimates are biased low. Dividing by `1 - beta^t` corrects the initialization bias.

3. **Q: How is Adam related to momentum?**
   **A:** Its first moment is momentum-like smoothing of gradients.

4. **Q: How is Adam related to RMSProp?**
   **A:** Its second moment normalizes updates by recent squared-gradient magnitude, similar to RMSProp.

5. **Q: What is AdamW?**
   **A:** Adam with decoupled weight decay. The decay is applied directly to parameters rather than through the adaptive gradient path.

6. **Q: Why is warmup common with AdamW?**
   **A:** Early moment estimates are based on little data and can be noisy. Warmup limits update size until training dynamics stabilize.

7. **Q: When might SGD with momentum beat Adam?**
   **A:** In well-tuned vision tasks or settings where final generalization beats fast training loss reduction.

8. **Q: What does epsilon do?**
   **A:** It prevents division by zero and affects update scale when `vhat` is very small.

9. **Q: Why can Adam struggle with generalization?**
   **A:** Adaptive scaling can change implicit regularization and may converge to sharper or less transferable solutions if not tuned carefully.

## Explain it like a principal

Adam is a practical preconditioner: it estimates recent gradient direction and scale using cheap diagonal statistics. That makes it forgiving across heterogeneous parameters, which is why it dominates large neural network training. The engineering maturity test is knowing when the defaults are insufficient: warmup for unstable starts, AdamW for clean weight decay, gradient clipping for spikes, schedule tuning for late-stage quality, and telemetry to distinguish optimization failure from data or model failure.

## Gotchas & follow-ups

- **"Adam's first step explodes because `v` is near zero."** With bias correction, `vhat` is correctly scaled for the first observed gradient. Instability is more often from noisy early gradients and too-large learning rates.
- **"AdamW is just Adam with a different name."** The decoupled decay changes regularization behavior.
- **"Lower beta values always adapt faster and are better."** They respond faster but are noisier.
- **Follow-up:** How would you tune learning rate, warmup, and weight decay together for a new transformer?
