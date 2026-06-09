# Derivatives and Partials — Interview Prep

## Q&A

1. **Q: What is the difference between a derivative and a partial derivative?**
   **A:** A derivative describes local change for a single-input function. A partial derivative describes local change of a multivariable function with respect to one coordinate while holding the other coordinates fixed. The gradient is the vector of all first-order partial derivatives for a scalar-valued function.

2. **Q: Why do derivatives matter for neural-network training?**
   **A:** Training minimizes a loss by changing parameters. The derivative of the loss with respect to each parameter gives the local sensitivity of the loss to that parameter, including sign and scale. Optimizers use those derivatives to update parameters more efficiently than search or random perturbation.

3. **Q: What does it mean for a derivative to be local?**
   **A:** It describes the slope at a point or in an infinitesimal neighborhood. It does not guarantee that the same slope holds after a large update. This is why learning rate matters: a gradient direction can be correct locally while a large step still overshoots.

4. **Q: How would you check a hand-derived gradient?**
   **A:** Compare it with autograd on random inputs and with centered finite differences over several step sizes. Autograd catches algebraic mistakes in manual derivations; finite differences catch mismatches between the intended mathematical function and the computation graph, though they are sensitive to numerical precision.

5. **Q: What causes finite-difference gradient checks to fail?**
   **A:** Too-large step sizes introduce truncation error because the approximation is no longer local. Too-small step sizes cause cancellation and roundoff error. Checks can also fail at nondifferentiable points, with stochastic layers, or when the function uses lower precision.

6. **Q: Are ReLU networks differentiable?**
   **A:** They are differentiable almost everywhere, but not at activation input zero. Frameworks choose a subgradient convention at the kink, commonly zero. This is sufficient for gradient-based training because exact kink hits are rare in continuous settings, but dead ReLUs can still create zero-gradient regions.

7. **Q: What does a zero partial derivative imply?**
   **A:** Locally, changing that coordinate alone has no first-order effect on the function. It does not necessarily mean the coordinate is globally irrelevant, because higher-order effects, interactions with other coordinates, or movement away from the current point may matter.

8. **Q: Why is scale important when interpreting partial derivatives?**
   **A:** A partial derivative is measured per unit of its input coordinate. If two coordinates use different units or natural scales, comparing raw derivative magnitudes can be misleading. Standardization, log transforms, or dimensionless elasticities can make sensitivity comparisons more meaningful.

## Explain it like a principal

Derivatives are the contract between the model, the loss, and the optimizer. At principal level, the important point is not just knowing derivative rules; it is knowing what gradient signals mean operationally. A derivative is a local approximation, so update size, parameter scaling, and numerical precision determine whether that signal is useful. Partial derivatives let large systems assign local responsibility across many parameters, but raw partials are only interpretable relative to coordinate scale and computation-graph structure. Good training diagnosis often starts with derivative telemetry: gradient norms, zero-gradient regions, exploding sensitivities, and mismatches between manual formulas and autograd.

## Gotchas & follow-ups

- **"A zero derivative means the model cannot improve."** Not necessarily. It may be a local flat point, a saturated activation, a coordinate-scale issue, or a point where first-order information is insufficient.
- **"Finite differences are ground truth."** They are an approximation. Their reliability depends on step size, dtype, smoothness, and deterministic evaluation.
- **"Partials can be compared directly."** Only if the coordinates have comparable units and scales. A derivative with respect to learning rate is not directly comparable to one with respect to dropout.
- **"ReLU is differentiable enough, so kinks do not matter."** Kinks are usually manageable, but dead ReLUs and subgradient conventions can materially affect training.
- **Follow-up:** How would you debug a custom loss whose autograd gradient disagrees with your derivation?
