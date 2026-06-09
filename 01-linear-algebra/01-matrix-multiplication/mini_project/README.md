# Matrix Multiplication — Mini-Project: Shape-Safe Linear Layer

Build a small educational linear-layer reference that makes matrix multiplication, shape checks, and gradients explicit.

## Goal

Create a script that implements the forward and backward pass for `Y = XW + b`, compares the result against PyTorch autograd, and reports shape and numerical checks. This project should make the matrix products in a dense layer visible rather than hidden inside `nn.Linear`.

## Dataset

Use synthetic tensors generated inside the script:

```python
import torch

torch.manual_seed(0)
n, d_in, d_out = 64, 8, 3
X = torch.randn(n, d_in)
true_W = torch.randn(d_in, d_out)
true_b = torch.randn(d_out)
y = X @ true_W + true_b + 0.05 * torch.randn(n, d_out)
```

No repository dataset is required.

## Implementation Tasks

Create a future file such as `mini_project/linear_layer_check.py` and implement:

1. A forward function that computes `Y = X @ W + b` and checks all shapes before multiplying.
2. Mean-squared error between predictions and y.
3. Manual gradients for W, b, and X using the formulas from the README.
4. The same computation with tensors using `requires_grad=True`.
5. A comparison table showing max absolute difference between manual and autograd gradients.
6. A failing-shape demonstration that catches a transposed or incorrectly sized W before PyTorch raises a lower-level error.

## Expected Workflow

After creating the script, run it from this topic directory:

```bash
uv run python mini_project/linear_layer_check.py
```

If the environment does not use `uv`, run the same script with the repository's active Python environment.

## Expected Outputs

The script should print:

- Input, weight, bias, and output shapes.
- Initial loss value.
- Max absolute gradient differences for X, W, and b.
- A clear message for the intentionally invalid shape case.

Gradient differences should be near floating-point tolerance, typically below `1e-6` for float32 on CPU.

## Writeup Prompt

Write 5-8 sentences explaining:

1. Why `dW = X^T G` has the same shape as W.
2. Why db sums over the batch dimension.
3. Which shape bug your checks caught.
4. How this exercise maps to `nn.Linear`.

## Optional Extensions

- Add a batched matrix multiplication example for attention scores.
- Compare a naive loop implementation against `torch.matmul`.
- Repeat the checks with float64 and compare numerical tolerances.
