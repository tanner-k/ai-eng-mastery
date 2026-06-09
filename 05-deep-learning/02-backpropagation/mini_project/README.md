# Backpropagation — Mini-Project: Manual MLP Backward Pass

## Goal

Build a two-layer neural network on synthetic data and implement the backward pass by hand. The project should verify that each manual gradient matches PyTorch autograd.

## Dataset

Generate a synthetic binary classification dataset inside the learner-created script:

- `X`: 256 samples with 4 features.
- Labels from a nonlinear rule such as `(x0 * x1 + x2 - x3 > 0)`.
- One hidden layer with 8 hidden units.

## Implementation Tasks

1. Initialize `W1`, `b1`, `W2`, and `b2`.
2. Run a forward pass with `tanh` or `relu` hidden activations.
3. Use a simple differentiable scalar loss.
4. Cache all forward values needed for backward.
5. Derive and implement manual gradients for every parameter.
6. Run the same computation with `requires_grad=True` and compare manual gradients to autograd.

## Expected Workflow

After creating a script such as `mini_project/manual_mlp_backprop.py`, the learner should be able to run:

```bash
uv run python mini_project/manual_mlp_backprop.py
```

This is a future command for the learner-created script; this content pass does not add that file.

## Expected Outputs

- Printed shape table for activations and gradients.
- Maximum absolute error between manual and autograd gradients for each parameter.
- A short training run showing the loss decreases after applying manual gradients.

## Writeup Prompt

Explain why the backward pass can compute all parameter gradients in one reverse sweep. Identify the cached value that would be most likely to cause a bug if omitted.

## Optional Extensions

- Add gradient checking with finite differences for a few selected parameters.
- Compare memory use when caching all activations versus recomputing hidden activations.
