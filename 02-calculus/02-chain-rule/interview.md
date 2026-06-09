# The Chain Rule — Interview Prep

## Q&A

1. **Q: What does the chain rule say in the context of model training?**
   **A:** It says the sensitivity of a loss to an early parameter is the product of local sensitivities along paths from that parameter to the loss, summed across paths when the computation branches. Backpropagation is an efficient bookkeeping system for those products and sums.

2. **Q: Why is the chain rule enough to train deep networks?**
   **A:** A network is a composition of differentiable or subdifferentiable primitive operations. If each primitive exposes a local derivative rule, reverse-mode autodiff can combine them to compute gradients of one scalar loss with respect to many parameters efficiently.

3. **Q: What is reverse-mode autodiff?**
   **A:** Reverse-mode autodiff computes derivatives from outputs back to inputs. For a scalar loss and many parameters, it is efficient because one backward pass computes all parameter gradients, reusing local derivatives recorded during the forward pass.

4. **Q: What happens when a computation graph branches?**
   **A:** Gradients from all downstream uses add together. If `x` affects the loss through two branches, the total derivative `dL/dx` is the sum of the derivative contribution from each branch.

5. **Q: How does the chain rule explain vanishing gradients?**
   **A:** Gradients passed through many layers involve products of local derivatives or Jacobians. If their magnitudes are often below one, the product shrinks exponentially with depth, leaving early layers with tiny update signals.

6. **Q: How does it explain exploding gradients?**
   **A:** If local derivatives or Jacobian singular values are often above one, repeated multiplication can amplify the gradient exponentially. This can produce unstable parameter updates, overflow, or `nan` losses.

7. **Q: Why is caching intermediate values useful in backpropagation?**
   **A:** Local derivative rules often depend on forward-pass values. For example, sigmoid's derivative can be written as `p(1 - p)`, where `p` was computed in the forward pass. Caching avoids recomputation and ensures the backward pass uses the exact forward values.

8. **Q: What is a common chain-rule mistake in loss derivations?**
   **A:** Dropping an inner derivative, such as differentiating `log(1 + exp(a))` as `exp(a)` instead of `exp(a)/(1 + exp(a))` times `da/dw`. Another common mistake is forgetting to sum gradient contributions from reused values.

## Explain it like a principal

The chain rule is the mechanism that turns compositional software into trainable models. At principal level, it is useful for more than derivations: it explains why architecture choices affect optimization. Residual connections shorten gradient paths, saturating activations shrink local derivatives, normalization changes local conditioning, and repeated Jacobian products create gradient pathologies. When training fails, the question is often not "did autograd work?" but "what local factors did the graph multiply together, and did they leave a usable signal?"

## Gotchas & follow-ups

- **"Backprop is different from the chain rule."** Backprop is the chain rule executed efficiently on a graph.
- **"Gradients only multiply along one path."** Branches add. Shared subexpressions must accumulate all downstream gradient contributions.
- **"Autograd removes the need to understand derivatives."** Autograd computes what the graph says, not what you intended. You still need chain-rule reasoning to debug wrong graphs and unstable training.
- **"Vanishing gradients are only about sigmoid."** Sigmoid saturation is one example. Repeated Jacobian products, poor initialization, and recurrent depth can cause the same issue.
- **Follow-up:** How would you design a tiny autodiff engine that handles both multiplication and reuse of a variable?
