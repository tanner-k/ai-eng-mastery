# Activations: tanh and ReLU — Interview Prep

## Q&A

1. **Q: Why do neural networks need activation functions?**
   **A:** Without nonlinear activations, any stack of linear layers is equivalent to a single linear layer.

2. **Q: What is the derivative of tanh?**
   **A:** `1 - tanh(x)^2`. It is largest near zero and approaches zero when tanh saturates near `-1` or `1`.

3. **Q: What is the derivative of ReLU?**
   **A:** It is `1` for positive inputs and `0` for negative inputs, with an arbitrary subgradient choice at zero.

4. **Q: Why did ReLU improve deep-network training?**
   **A:** On active positive paths, its derivative is one, which preserves gradient flow better than saturated sigmoid or tanh units.

5. **Q: What is a dead ReLU?**
   **A:** A unit whose preactivation is negative for nearly all inputs, so its output and gradient are zero and it stops updating meaningfully.

6. **Q: When might tanh still be useful?**
   **A:** When bounded, zero-centered outputs are desirable, such as certain recurrent transformations or constrained output ranges.

7. **Q: Why is zero-centered activation sometimes helpful?**
   **A:** It can make optimization easier by avoiding consistently positive activations that bias gradient directions.

8. **Q: What are common ReLU variants?**
   **A:** Leaky ReLU, PReLU, ELU, GELU, and SiLU. They reduce dead-unit risk or provide smoother behavior.

9. **Q: How does activation choice affect initialization?**
   **A:** The activation changes variance propagation. ReLU commonly pairs with He initialization; tanh commonly pairs with Xavier initialization.

10. **Q: Are activations only about forward expressiveness?**
    **A:** No. Their derivatives shape the backward signal and therefore the trainability of the model.

## Explain it like a principal

Activation choice is an architecture and optimization decision at the same time. It determines what functions the network can represent, how variance moves forward, how gradients move backward, and what telemetry to monitor. In production, a principal engineer should be able to connect dead units, saturation, initialization, normalization, and learning-rate sensitivity into one explanation of trainability rather than treating activation functions as interchangeable nonlinearities.

## Gotchas & follow-ups

- **"More nonlinear is always better."** Saturating nonlinearities can block gradients.
- **Ignoring the derivative.** The backward behavior is often more important than the forward curve.
- **Treating ReLU's zero derivative as harmless.** Dead ReLUs can remove capacity.
- **Forgetting initialization.** Activation and initialization should be selected together.
- **Assuming tanh is obsolete.** It is less common in deep feed-forward blocks but still useful in bounded-state designs.

Follow-up: If the first three layers of a network stop learning, what activation, initialization, and telemetry checks would you run first?
