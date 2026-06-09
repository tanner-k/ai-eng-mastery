# Gradients and Jacobians — Interview Prep

## Q&A

1. **Q: What is the difference between a gradient and a Jacobian?**
   **A:** A gradient is the vector of partial derivatives for a scalar-valued function. A Jacobian is the matrix of partial derivatives for a vector-valued function. A gradient is a special case of derivative structure when the output dimension is one.

2. **Q: What direction does the gradient point?**
   **A:** Under the Euclidean norm, it points in the direction of steepest local increase of the scalar function. The negative gradient is the steepest local decrease direction, which is why optimizers step opposite the gradient when minimizing loss.

3. **Q: What is a directional derivative?**
   **A:** It is the local rate of change of a scalar function in a chosen direction `u`, computed as `grad f(x)^T u`. It projects the gradient onto the direction of interest.

4. **Q: Why do deep-learning frameworks often avoid forming full Jacobians?**
   **A:** Full Jacobians are usually too large. Training a scalar loss only needs gradients with respect to parameters, so reverse-mode autodiff computes vector-Jacobian products without materializing the entire matrix.

5. **Q: What is a vector-Jacobian product?**
   **A:** A VJP multiplies an upstream row-vector sensitivity by a local Jacobian to produce downstream sensitivities. It is the core operation in reverse-mode backpropagation.

6. **Q: What is a Jacobian-vector product?**
   **A:** A JVP multiplies a local Jacobian by an input perturbation vector to propagate perturbations forward. It is common in forward-mode autodiff, sensitivity analysis, and some second-order methods.

7. **Q: Why is the softmax Jacobian not diagonal?**
   **A:** Each softmax output depends on every input logit through the normalization denominator. Increasing one logit increases its probability but decreases the others, so the output sensitivities are coupled.

8. **Q: How do Jacobians relate to vanishing or exploding gradients?**
   **A:** Backpropagated gradients are multiplied by layer Jacobians. If the relevant singular values are repeatedly below one, gradients vanish; if they are repeatedly above one, gradients explode.

## Explain it like a principal

Gradients and Jacobians are the shape-aware language of learning systems. A principal engineer should reason about them as linear maps, not just derivative tables. That perspective explains why some operations are cheap to differentiate, why full Jacobians are infeasible for large vocabularies or images, and why architecture changes alter gradient flow. It also prevents shape mistakes: every backward signal must contract with a local derivative object into the shape of the upstream input or parameter.

## Gotchas & follow-ups

- **"The Jacobian of a batch always mixes examples."** Usually false. Standard feed-forward layers operate independently per example; the batch dimension is often block-diagonal conceptually. Batch normalization and attention can introduce cross-example or cross-token coupling depending on configuration.
- **"A gradient is always a column vector."** Convention varies. What matters is consistent shape algebra.
- **"Full Jacobian is needed for backprop."** Training usually needs VJPs, not materialized Jacobians.
- **"Largest gradient component is always the best feature to change."** Raw components depend on coordinate scale and constraints.
- **Follow-up:** When would forward-mode autodiff be preferable to reverse mode?
