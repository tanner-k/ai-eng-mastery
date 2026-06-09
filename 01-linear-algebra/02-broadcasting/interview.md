# Broadcasting — Interview Prep

## Q&A

1. **Q: What are the broadcasting rules?**
   **A:** Align shapes from the right. Each aligned dimension is compatible if the sizes are equal or one size is 1. Missing leading dimensions behave like size 1. The output uses the maximum size per dimension.

2. **Q: Why does adding a bias vector to a batch of activations work?**
   **A:** Activations might have shape `(batch, hidden)` and bias has shape `(hidden,)`. The bias is treated as `(1, hidden)` and reused for every batch row.

3. **Q: What happens to gradients through a broadcasted dimension?**
   **A:** The backward pass sums gradients over every axis where the input was broadcast. If one bias element contributed to 128 outputs, its gradient is the sum of those 128 upstream gradients.

4. **Q: Is broadcasting the same as copying data?**
   **A:** Not usually. Broadcasting can be represented as a view with adjusted strides, especially with `expand`. `repeat` materializes copies and uses more memory.

5. **Q: How can broadcasting create silent bugs?**
   **A:** A shape can be mechanically compatible but semantically wrong. For example, accidentally using `(batch, 1)` instead of `(1, features)` may scale each example rather than each feature.

6. **Q: How do you make broadcasting intent clear in production code?**
   **A:** Use explicit `view`, `reshape`, `unsqueeze`, or named shape comments near critical operations. Add assertions for high-risk axes such as batch, token, class, and head.

7. **Q: Why does stable softmax use broadcasting?**
   **A:** It subtracts the maximum value along a dimension while keeping that dimension as size 1, then broadcasts the max back across the original tensor before exponentiation.

8. **Q: What is the difference between `expand` and `repeat` in PyTorch?**
   **A:** `expand` returns a view and can only expand dimensions of size 1. `repeat` copies data and can duplicate any dimension, but it increases memory use.

9. **Q: How does broadcasting interact with batch and head dimensions in attention?**
   **A:** Masks or biases are often shaped with singleton dimensions such as `(batch, 1, 1, tokens)` so they can apply across all heads and query positions.

10. **Q: When should you avoid broadcasting?**
    **A:** Avoid implicit broadcasting when axes are ambiguous, when a wrong shape would still be compatible, or when future maintainers need explicit semantics more than compact code.

## Explain it like a principal

Broadcasting is a shape contract, not just syntactic convenience. In large model code, it controls whether a bias, mask, statistic, or loss weight is applied per token, per class, per feature, per head, or per example. Principal-level review should focus on axis semantics: which dimensions are intentionally singleton, which are preserved, and which reductions will happen in backward. The difference between a correct training run and a subtly corrupted one is often a single missing `unsqueeze`.

## Gotchas & follow-ups

- **Right-alignment surprises.** `(batch,)` does not automatically mean "per batch item" when combined with `(batch, classes)`.
- **Gradient reduction omissions.** Manual backward code must sum over broadcasted axes, or parameter gradients will have the wrong shape and magnitude.
- **`expand` view constraints.** Expanded dimensions may have stride 0; in-place writes to expanded views are dangerous or disallowed.
- **Mask shape ambiguity.** In attention, explain whether a mask applies per batch, per head, per query, or per key.
- **Follow-up prompt:** Given logits `(B, T, C)` and class weights `(C,)`, per-token weights `(B, T)`, and padding mask `(B, T)`, reshape each one for correct broadcasting.
