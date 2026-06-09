# Derivatives and Partials — Mini-Project: Gradient Check Lab

Build a small gradient-checking lab for scalar functions and a one-example linear-model loss.

---

## Goal

Create a script that compares manual derivatives, centered finite differences, and PyTorch autograd. The project should make numerical derivative checking feel concrete rather than magical.

---

## Data Setup

Use synthetic scalar inputs and small tensors generated inside the script. No external data is needed.

Suggested cases:

- `f(x) = 3x^2 - 2x + 5`
- `g(x) = sin(x) exp(-x)`
- `L(w, b) = (w^T x + b - y)^2` for `x = [2.0, -1.0, 0.5]`
- `r(x) = max(0, x)` evaluated near `x = 0`

---

## Implementation Tasks

1. Create a future script such as `mini_project/gradient_check_lab.py`.
2. Implement centered finite differences for scalar inputs.
3. Write manual derivatives for the smooth functions.
4. Use `torch.autograd` to compute the same derivatives.
5. Sweep `h` values such as `1e-1`, `1e-3`, `1e-5`, `1e-7`, and `1e-9`.
6. Print a table with manual, finite-difference, autograd, and absolute-error columns.
7. Include one nondifferentiable ReLU example and explain why the check is ambiguous at zero.

---

## Expected Workflow

After creating the script, a learner should be able to run it from this topic directory with a command like:

```bash
uv run python mini_project/gradient_check_lab.py
```

If the repository environment is not installed, running the same file with a Python environment that has PyTorch available is sufficient.

---

## Expected Outputs

- A printed table showing that smooth functions match autograd for reasonable `h`.
- A short note identifying where finite differences fail because `h` is too large or too small.
- A ReLU-at-zero example showing that nondifferentiability makes "the" derivative convention-dependent.

---

## Writeup Prompt

Write 5-7 sentences explaining which derivative check you would trust in production and why. Include one concrete failure mode for finite differences and one failure mode for autograd-based checks.

---

## Optional Extensions

- Add vector inputs and compare each coordinate's partial derivative.
- Repeat the experiment in float32 and float64.
- Add random test cases and assert that errors stay below a tolerance for smooth functions.
