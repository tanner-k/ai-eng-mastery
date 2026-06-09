# Batch Normalization — Interview Prep

## Q&A

1. **Q: What does batch normalization do?**
   **A:** It normalizes activations using mini-batch mean and variance, then applies learned scale `gamma` and shift `beta`.

2. **Q: Why include `gamma` and `beta`?**
   **A:** They let the network recover any needed scale or offset after normalization, preserving representational flexibility.

3. **Q: What is different between training and inference?**
   **A:** Training uses current mini-batch statistics. Inference uses running estimates accumulated during training for deterministic behavior.

4. **Q: Why can batch norm allow larger learning rates?**
   **A:** It stabilizes activation scale and can smooth optimization, reducing sensitivity to parameter updates in earlier layers.

5. **Q: What axes does BatchNorm2d normalize over?**
   **A:** For NCHW tensors, it computes statistics per channel over batch, height, and width.

6. **Q: Why is batch norm problematic with tiny batches?**
   **A:** The mean and variance estimates are noisy, so normalized activations fluctuate and running estimates may be poor.

7. **Q: How does batch norm interact with convolution bias?**
   **A:** A bias before batch norm is often redundant because batch norm subtracts the batch mean and then has its own learned shift.

8. **Q: What is synchronized batch norm?**
   **A:** It computes batch statistics across devices rather than per device, giving larger effective batch statistics in distributed training.

9. **Q: Why might batch norm hurt sequence models or transformers?**
   **A:** It depends on batch statistics and can mix sequence/batch assumptions. Layer norm is usually preferred because it normalizes within each example.

10. **Q: What should you check when fine-tuning a batch-norm model?**
    **A:** Whether batch-norm parameters and running statistics are frozen or updated, and whether the fine-tuning batch distribution is representative.

## Explain it like a principal

Batch norm is not only a mathematical transform; it is stateful training infrastructure. Its behavior depends on mode, batch composition, distributed topology, and running-statistics policy. Principal-level decisions include whether the batch size supports reliable estimates, whether synchronized stats are needed, how to fine-tune pretrained models, and whether an alternative normalization better matches the architecture. Many deployment bugs come from a model accidentally left in training mode or from running statistics that no longer match production data.

## Gotchas & follow-ups

- **Forgetting eval mode.** Inference in training mode produces batch-dependent outputs.
- **Assuming momentum means the same thing in every library.** Framework conventions differ.
- **Using tiny batches without question.** Batch statistics may be too noisy.
- **Leaving redundant biases.** Bias before batch norm is usually unnecessary.
- **Confusing batch norm and layer norm.** Batch norm normalizes across batch examples; layer norm normalizes within an example.

Follow-up: During fine-tuning, when would you freeze batch-norm running statistics, and what validation signal would tell you that choice is wrong?
