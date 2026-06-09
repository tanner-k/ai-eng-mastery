# Convolutional Neural Networks — Interview Prep

## Q&A

1. **Q: What is a convolutional layer?**
   **A:** It applies learned local filters across spatial positions, producing output feature maps through weight sharing.

2. **Q: Why are CNNs parameter-efficient for images?**
   **A:** A small kernel is reused at every location instead of learning separate weights for every pixel-position pair.

3. **Q: What is translation equivariance?**
   **A:** If the input shifts, the convolutional feature map shifts correspondingly, assuming boundary effects are ignored.

4. **Q: What is a receptive field?**
   **A:** The region of the input that can affect a particular output unit.

5. **Q: How do stride and padding affect output shape?**
   **A:** Stride skips positions and reduces resolution. Padding adds border values and can preserve spatial size.

6. **Q: What is the difference between channels and spatial dimensions?**
   **A:** Channels represent feature types; spatial dimensions represent positions. Kernels mix input channels locally across space.

7. **Q: Why use pooling?**
   **A:** Pooling reduces spatial resolution and can add local invariance, though strided convolutions are often used instead.

8. **Q: How does convolution relate to matrix multiplication?**
   **A:** Input patches can be lowered into columns with `im2col`, making convolution equivalent to a matrix multiply plus reshaping.

9. **Q: Why are `1x1` convolutions useful?**
   **A:** They mix channels independently at each spatial location, often reducing or expanding channel dimension cheaply.

10. **Q: What are common CNN debugging checks?**
    **A:** Verify tensor layout, output shapes, receptive-field size, activation statistics, and whether downsampling happens too early.

## Explain it like a principal

CNNs encode a prior: local patterns repeat across space. The engineering value is not only fewer parameters; it is better sample efficiency and predictable control over resolution, receptive field, and compute. Principal-level design means knowing where to spend spatial resolution, how fast to downsample, when to increase channel width, and how normalization and activation choices affect training. Many CNN failures are shape or resolution failures: the model downsamples before preserving the information the task needs, or it never gets a receptive field large enough to see the relevant context.

## Gotchas & follow-ups

- **Confusing equivariance and invariance.** Convolution is equivariant; pooling or global aggregation can create partial invariance.
- **Ignoring tensor layout.** NCHW and NHWC mismatches produce bugs or performance issues.
- **Downsampling too early.** Small objects or fine details can disappear.
- **Assuming bigger kernels are always better.** Stacked small kernels can grow receptive field with fewer parameters and more nonlinearities.
- **Forgetting bias redundancy.** Bias before batch norm is often unnecessary.

Follow-up: Given a CNN that misses small objects, what shape, stride, receptive-field, and feature-map diagnostics would you inspect before changing the model family?
