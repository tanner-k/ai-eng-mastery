# Convolutional Neural Networks — Mini-Project: Synthetic Shape Classifier

## Goal

Build a small CNN that classifies synthetic image patterns and use it to study kernels, downsampling, and receptive fields.

## Dataset

Generate images inside the script:

- Grayscale images with shape `(1, 28, 28)`.
- Three classes such as vertical bar, horizontal bar, and diagonal bar.
- Random position, thickness, and noise added to each image.
- Train/validation split generated from separate random seeds.

No image files or external datasets are required.

## Implementation Tasks

1. Generate synthetic images and integer labels.
2. Build a small CNN with convolution, activation, optional batch norm, and pooling or stride.
3. Train the model and report validation accuracy.
4. Print output shapes after each layer.
5. Compare stride `1` plus pooling against an early stride `2` convolution.
6. Compute or print the receptive field size for the final feature map.

## Expected Workflow

After creating a script such as `mini_project/synthetic_shape_cnn.py`, the learner should be able to run:

```bash
uv run python mini_project/synthetic_shape_cnn.py
```

This is a future command for learner-created code; no script is added during this content pass.

## Expected Outputs

- Validation accuracy for each architecture variant.
- Layer-by-layer shape table.
- Parameter count comparison.
- A short analysis of downsampling choices.

## Writeup Prompt

Which architecture handled the synthetic patterns best? Explain using receptive field, output shapes, and whether early downsampling removed useful spatial detail.

## Optional Extensions

- Add color channels and class-specific color cues.
- Implement a naive convolution for one layer and compare it to PyTorch.
- Visualize learned first-layer filters if the learner adds plotting.
