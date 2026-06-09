# Backpropagation — Interview Prep

## Q&A

1. **Q: What is backpropagation?**
   **A:** Backpropagation is reverse-mode differentiation over a computation graph. It computes gradients of one scalar loss with respect to many parameters by propagating vector-Jacobian products backward.

2. **Q: Why is reverse mode preferred for neural-network training?**
   **A:** Training has many inputs to differentiate with respect to and usually one scalar loss. Reverse mode computes all parameter gradients in one backward sweep, while forward mode would be efficient for few inputs and many outputs.

3. **Q: What does a layer need to implement a backward pass?**
   **A:** It needs the upstream gradient and the local forward values needed by its derivative, such as inputs, weights, activation prevalues, or masks.

4. **Q: Why are full Jacobians usually avoided?**
   **A:** They are enormous and mostly unnecessary. Backprop only needs products between upstream gradients and local Jacobians, which can be computed directly with matrix operations.

5. **Q: How do bias gradients arise in an affine layer?**
   **A:** The same bias is added to every batch row, so the bias gradient is the sum of upstream gradients over the batch dimension.

6. **Q: How does backprop relate to dynamic programming?**
   **A:** It reuses gradients of downstream subgraphs instead of recomputing the effect of each parameter on the loss independently.

7. **Q: What is the difference between `dL/dz` and `dL/da` in an activation layer?**
   **A:** `dL/da` is the upstream gradient with respect to activation output. `dL/dz` multiplies that by the activation derivative evaluated at the preactivation.

8. **Q: Why can in-place operations break autograd?**
   **A:** The backward pass may need the original forward value. If it is overwritten in-place, the stored computation graph no longer has the data required for a correct derivative.

9. **Q: What is gradient checkpointing?**
   **A:** It trades compute for memory by discarding selected activations in the forward pass and recomputing them during backward.

10. **Q: What makes backprop numerically unstable in very deep networks?**
    **A:** Repeated multiplication by Jacobians can shrink or amplify signals exponentially, causing vanishing or exploding gradients.

## Explain it like a principal

Backpropagation is the training system's dataflow contract. Every operation must define how upstream sensitivity maps to its inputs and parameters, and the runtime schedules those local rules in reverse topological order. Principal-level reasoning focuses on memory, graph boundaries, nondifferentiable pieces, mixed precision, and whether the gradients being optimized actually correspond to the intended loss. When debugging a training failure, gradient telemetry is often more informative than the loss curve alone: norms by layer, zero-gradient checks, exploding steps, and unexpected `None` gradients tell you where the chain rule stopped being useful.

## Gotchas & follow-ups

- **"Backprop is gradient descent."** Backprop computes gradients; gradient descent uses them to update parameters.
- **"It computes derivatives layer by layer independently."** Local derivatives are independent, but their contribution to the loss depends on upstream gradients.
- **Forgetting caches.** Without forward activations, many backward equations cannot be evaluated.
- **Confusing reverse and forward mode.** Ask which mode is efficient for a scalar loss and millions of parameters.
- **Ignoring reduction semantics.** Mean losses, batch sums, and broadcasted parameters all change gradient scale.
