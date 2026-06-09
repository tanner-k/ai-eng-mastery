# Broadcasting in Neural Networks — Interview Prep

## Q&A

1. **Q: What is broadcasting in tensor libraries?**
   **A:** Broadcasting lets tensors with different but compatible shapes participate in elementwise operations by treating dimensions of size `1` or missing leading dimensions as repeated across the larger tensor. The repetition is usually virtual, not a physical copy.

2. **Q: Why does broadcasting matter in neural networks?**
   **A:** It is the mechanism behind bias addition, normalization parameters, masks, per-channel scales, and many loss computations. A wrong broadcast can silently change the mathematical problem while still producing a tensor and a scalar loss.

3. **Q: How are shapes aligned for broadcasting?**
   **A:** Shapes are compared from the right. Each aligned dimension must be equal or one of them must be `1`. Missing leading dimensions behave like size `1`.

4. **Q: What happens to gradients through a broadcasted tensor?**
   **A:** The backward pass sums the upstream gradient over every axis along which the original tensor was broadcast. If a bias vector is used for every batch row, its gradient is the sum over batch rows.

5. **Q: Why is `(B, 1) - (B,)` dangerous?**
   **A:** Trailing alignment treats `(B,)` as `(1, B)`, so the result becomes `(B, B)`. Instead of comparing each prediction with its matching target, it compares every prediction with every target.

6. **Q: Is `expand` the same as `repeat`?**
   **A:** No. `expand` creates a view with virtual repetition, often using stride `0`; `repeat` materializes copied values. `expand` is memory efficient, but some downstream operations may require contiguous materialized storage.

7. **Q: How would you apply a channel-wise parameter to an NCHW image tensor?**
   **A:** Reshape the parameter from `(C,)` to `(1, C, 1, 1)` so it broadcasts across batch, height, and width while matching channels.

8. **Q: What shape checks would you add in production model code?**
   **A:** Assert expected rank and key dimensions at layer boundaries, normalize target shapes before losses, and prefer explicit `view` or `unsqueeze` calls for nontrivial broadcasts.

9. **Q: Can broadcasting hide data leakage or target bugs?**
   **A:** Yes. A wrong target shape can turn per-example losses into pairwise matrices, mixing examples inside a batch and producing misleadingly smooth losses.

10. **Q: How does broadcasting interact with parameter sharing?**
    **A:** Broadcasting reuses one parameter value across many positions. That is parameter sharing, and the gradient aggregates all positions that used the shared value.

## Explain it like a principal

Broadcasting is part of the model contract. At principal level, you should reason about every singleton dimension as a statement of sharing: same bias for all examples, same scale for all pixels in a channel, same mask across heads, or same scalar temperature for every logit. The forward pass is only half of the contract; the backward pass must reduce gradients back to the parameter's true shape. The production risk is that broadcasting errors often do not crash. They create plausible losses with wrong semantics, especially in training loops where targets, logits, masks, and weights meet.

## Gotchas & follow-ups

- **"Broadcasting copies the smaller tensor."** Usually false. Frameworks often create views. Ask how stride `0` works and when contiguity matters.
- **Ignoring trailing alignment.** Candidates often align shapes from the left. Follow up with `(B, 1) - (B,)`.
- **Forgetting gradient reduction.** If the same bias is used for every row, the bias gradient cannot be one row's gradient; it is the sum across rows.
- **Overusing implicit broadcasting.** Explicit `unsqueeze` improves readability when multiple singleton axes are involved.
- **Assuming a passing forward pass means correct math.** Broadcasting bugs often pass type and shape checks unless the checks encode intended semantics.
