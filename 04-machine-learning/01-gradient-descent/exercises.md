# Gradient Descent — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Derive the MSE gradient by hand

Given the MSE loss for linear regression:

```
L(w, b) = (1/n) · Σᵢ (xᵢᵀw + b − yᵢ)²
```

Derive ∂L/∂w and ∂L/∂b by hand using the chain rule. Express your answer in matrix form for the full dataset X ∈ ℝⁿˣᵈ, y ∈ ℝⁿ. Verify your derivation matches the implementation in `gd/models.py:LinearModel.gradients()`.

---

## Exercise 2 — Implement Adam from scratch and match PyTorch

Using only the update equations in the README (no peeking at `gd/optimizers.py`), implement an `Adam` class with a `step(params, grads) -> list[Tensor]` signature. Then:

1. Train a `LinearModel` on the synthetic dataset below for 200 steps using your Adam.
2. Train the same model using `torch.optim.Adam` with identical hyperparameters.
3. Assert that the final `w` and `b` from both runs agree within `atol=1e-2`.

```python
import torch

torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([1.5, -2.0, 0.7])
true_b = torch.tensor(0.3)
y = X @ true_w + true_b + 0.1 * torch.randn(100)
```

Refer to `[[adam-optimization]]` for the full update equations if needed.

---

## Exercise 3 — Build a learning-rate schedule and observe its effect

Implement one of the following schedules (your choice) and compare convergence to a constant learning rate:

- **Step decay**: halve the learning rate every 50 steps.
- **Cosine annealing**: decay from `lr_max` to `lr_min` following `lr(t) = lr_min + 0.5·(lr_max − lr_min)·(1 + cos(πt/T))`.

Use `gd.optimizers.SGD` on the synthetic dataset from Exercise 2. Plot (or print) the loss at each step for both the constant-rate and scheduled runs. Write two to three sentences explaining what you observe: does the schedule converge to a lower loss, converge faster, or show different late-stage behavior?

---

## Exercise 4 — Construct a diverging learning rate

Choose a learning rate large enough to make training diverge (loss increases on every step or becomes `nan`). Run `SGD` with this rate on the synthetic dataset from Exercise 2 and print the first 20 loss values to confirm divergence. Then explain:

1. Why does this specific learning rate cause divergence? Relate your answer to the Lipschitz constant of the MSE gradient.
2. How would you diagnose divergence in a real training run, and what are two immediate remedies?

---

## Exercise 5 — Vectorize a per-sample loop and benchmark the speedup

The naive per-sample training loop below computes gradients one sample at a time:

```python
from gd.models import LinearModel
from gd.optimizers import SGD
import torch
import time

torch.manual_seed(0)
X = torch.randn(1000, 10)
y = X @ torch.ones(10) + torch.randn(1000) * 0.1

model = LinearModel(w=torch.zeros(10), b=torch.tensor(0.0))
opt = SGD(lr=0.01)

start = time.perf_counter()
for _ in range(50):          # epochs
    for i in range(len(X)):  # per-sample loop
        xi = X[i:i+1]
        yi = y[i:i+1]
        gw, gb = model.gradients(xi, yi)
        model_w, model_b = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=model_w, b=model_b)
naive_time = time.perf_counter() - start
print(f"Naive: {naive_time:.3f}s, final loss: {model.loss(X, y):.4f}")
```

Rewrite the inner loop as a single mini-batch gradient step over all 1000 samples per epoch (batch GD). Time both versions and report the wall-clock speedup. Explain why vectorization is faster even though both versions perform the same number of arithmetic operations.
