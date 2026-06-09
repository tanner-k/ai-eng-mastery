# Matrix Multiplication — Interview Prep

## Q&A

1. **Q: What does matrix multiplication compute?**
   **A:** For A in R^(m x n) and B in R^(n x p), AB is an `(m, p)` matrix whose `(i, j)` entry is the dot product between row i of A and column j of B. It computes many related dot products in one structured operation.

2. **Q: Why is matrix multiplication central to neural networks?**
   **A:** Dense layers, attention projections, attention score matrices, many convolution implementations, and output classifiers are all matrix products. Training also depends on matrix products in the backward pass, such as `X^T @ G` for weight gradients.

3. **Q: Is matrix multiplication commutative?**
   **A:** No. Usually `AB != BA`, and often one product is valid while the other is not. Multiplication represents composition of transformations, and changing order changes the transformation.

4. **Q: What is the shape of `X @ W + b` for X `(batch, d_in)`, W `(d_in, d_out)`, and b `(d_out,)`?**
   **A:** `X @ W` has shape `(batch, d_out)`. The bias broadcasts across the batch dimension, so the final output is also `(batch, d_out)`.

5. **Q: Derive the gradient of `Y = XW` with respect to W.**
   **A:** If G = dL/dY, then dL/dW = X^T G. The shape is `(d_in, batch) @ (batch, d_out) -> (d_in, d_out)`, matching W.

6. **Q: Why does attention use `Q @ K^T`?**
   **A:** Each query token needs a score against each key token. With Q and K shaped by token and head dimension, transposing K produces a `(query_tokens, key_tokens)` score matrix per batch and head.

7. **Q: What is the difference between elementwise multiplication and matrix multiplication?**
   **A:** Elementwise multiplication multiplies corresponding entries and requires broadcast-compatible shapes. Matrix multiplication contracts one dimension with a sum of products. Confusing them changes both shape and semantics.

8. **Q: Why can parenthesization affect performance even when the result is mathematically the same?**
   **A:** Matrix multiplication is associative, so valid parenthesizations return the same result, but intermediate shapes can differ dramatically. Smaller intermediates reduce compute and memory traffic.

9. **Q: How do batched matrix products generalize ordinary matrix multiplication?**
   **A:** The last two dimensions are treated as matrix dimensions, and leading dimensions are batch dimensions. Broadcast-compatible leading dimensions allow many matrix products to run in one call.

10. **Q: What numerical issues matter for large matrix products?**
    **A:** Floating-point summation order, precision, scaling, and conditioning can all affect results. In deep learning, mixed precision matmul is fast but requires care around accumulation precision and loss scaling.

## Explain it like a principal

Matrix multiplication is the contract between model math and hardware. At principal level, you should be able to look at a proposed architecture and infer the matrix products, tensor shapes, memory footprint, and gradient paths before code is written. This is how you catch impossible attention shapes, excessive activation memory, accidental quadratic costs, and parameter matrices with the wrong orientation. The concept is simple, but production failures often come from treating it as syntax instead of as a precise operation with shape, cost, and numerical consequences.

## Gotchas & follow-ups

- **"The output shape is whatever PyTorch accepts."** Shape acceptance is not semantic correctness. Broadcasting or transposes can make code run while computing the wrong quantity.
- **Forgetting PyTorch weight orientation.** `nn.Linear(in, out)` stores weight as `(out, in)` and applies `input @ weight.T + bias`.
- **Confusing associativity with commutativity.** `(AB)C = A(BC)` when valid, but `AB` generally does not equal `BA`.
- **Ignoring intermediate size.** A mathematically valid product order can be unusable because it materializes a huge intermediate.
- **Attention follow-up:** Given Q, K, V shapes, explain every dimension of `softmax(QK^T / sqrt(d))V`, not just the final output.
