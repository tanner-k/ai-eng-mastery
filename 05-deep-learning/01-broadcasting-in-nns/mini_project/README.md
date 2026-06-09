# Broadcasting in Neural Networks — Mini-Project: Shape Bug Lab

## Goal

Build a small PyTorch script that demonstrates correct and incorrect broadcasting in neural-network training code. The deliverable will be a script plus a short analysis of how each shape choice changes the loss and gradients.

## Dataset

Use synthetic tensors generated inside the script:

- Regression inputs `X` with shape `(64, 3)`.
- Targets `y` created from a known linear function with shape `(64,)`.
- Image-like tensor `imgs` with shape `(8, 3, 16, 16)` for per-channel scaling.

No repository dataset is required.

## Implementation Tasks

1. Create a linear prediction `pred = X @ w + b` and intentionally compare both `(64,)` predictions and `(64, 1)` predictions against `(64,)` targets.
2. Print the intermediate loss tensor shapes before reduction.
3. Manually compute the bias gradient as a sum over the batch and compare it to `autograd`.
4. Apply channel-wise `gamma` to `imgs` by reshaping it to `(1, 3, 1, 1)`.
5. Compare `expand` and `repeat` by printing resulting shapes, strides, and memory behavior.

## Expected Workflow

After creating a script such as `mini_project/shape_bug_lab.py`, the learner should be able to run:

```bash
uv run python mini_project/shape_bug_lab.py
```

This command is a future workflow for the learner-created script; the file is not provided in this content pass.

## Expected Outputs

- A table of valid and invalid broadcast examples.
- Printed evidence that `(64, 1) - (64,)` becomes `(64, 64)`.
- Matching manual and autograd bias gradients.
- A short comparison of `expand` versus `repeat`.

## Writeup Prompt

Explain which broadcasting pattern was most dangerous and why it would be easy to miss during model training. Include one shape assertion you would add before computing a loss.

## Optional Extensions

- Add a tiny attention-mask example with shapes `(B, 1, 1, T)` and `(B, H, T, T)`.
- Benchmark `expand` versus `repeat` on a larger tensor and report peak memory if your environment exposes it.
