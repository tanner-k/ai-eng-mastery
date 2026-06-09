# Gradient Descent — Solutions

---

## Solution 1 — MSE gradient derivation

### Setup

Let predictions be ŷ = Xw + b where X ∈ ℝⁿˣᵈ, w ∈ ℝᵈ, b ∈ ℝ, y ∈ ℝⁿ. The MSE loss is:

```
L(w, b) = (1/n) · ‖Xw + b − y‖²
         = (1/n) · (Xw + b − y)ᵀ(Xw + b − y)
```

Let r = Xw + b − y (residual vector, r ∈ ℝⁿ).

### Gradient with respect to w

Expand using the chain rule. Let f = rᵀr / n:

```
∂L/∂w = (1/n) · ∂(rᵀr)/∂w
       = (1/n) · 2rᵀ · ∂r/∂w
```

The Jacobian ∂r/∂w = X (since r = Xw + b − y, only the Xw term depends on w):

```
∂L/∂w = (2/n) · Xᵀr
       = (2/n) · Xᵀ(Xw + b − y)
```

### Gradient with respect to b

```
∂L/∂b = (1/n) · 2rᵀ · ∂r/∂b
```

Since ∂r/∂b = 1 (an n-vector of ones):

```
∂L/∂b = (2/n) · Σᵢ rᵢ
       = (2/n) · (Xw + b − y).sum()
```

### Verification against gd/models.py

These match exactly:

```python
# gd/models.py lines 29-32
residual = self.predict(X) - y          # r = Xw + b - y
grad_w = (2.0 / n) * (X.T @ residual)  # (2/n) Xᵀr
grad_b = (2.0 / n) * residual.sum()    # (2/n) Σ rᵢ
```

---

## Solution 2 — Adam from scratch vs. torch.optim.Adam

```python
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from gd.models import LinearModel

# ── Dataset ─────────────────────────────────────────────────────────────────
torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([1.5, -2.0, 0.7])
true_b = torch.tensor(0.3)
y = X @ true_w + true_b + 0.1 * torch.randn(100)

# ── From-scratch Adam ────────────────────────────────────────────────────────
@dataclass
class AdamScratch:
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: list[torch.Tensor] | None = None
    v: list[torch.Tensor] | None = None
    t: int = 0

    def step(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if self.m is None:
            self.m = [torch.zeros_like(p) for p in params]
            self.v = [torch.zeros_like(p) for p in params]
        self.t += 1
        new_params, new_m, new_v = [], [], []
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            new_params.append(p - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps))
            new_m.append(m)
            new_v.append(v)
        self.m, self.v = new_m, new_v
        return new_params


def train_scratch(steps: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    model = LinearModel(w=torch.zeros(3), b=torch.tensor(0.0))
    opt = AdamScratch(lr=0.01)
    for _ in range(steps):
        gw, gb = model.gradients(X, y)
        new_w, new_b = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=new_w, b=new_b)
    return model.w, model.b


# ── PyTorch Adam ─────────────────────────────────────────────────────────────
def train_torch(steps: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros(3, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((X @ w + b - y) ** 2)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


scratch_w, scratch_b = train_scratch()
torch_w, torch_b = train_torch()

assert torch.allclose(scratch_w, torch_w, atol=1e-2), (
    f"w mismatch: {scratch_w} vs {torch_w}"
)
assert torch.allclose(scratch_b, torch_b.squeeze(), atol=1e-2), (
    f"b mismatch: {scratch_b} vs {torch_b}"
)
print("Adam scratch vs torch.optim.Adam: PASS")
print(f"  scratch w={scratch_w.tolist()}, b={scratch_b.item():.4f}")
print(f"  torch   w={torch_w.tolist()}, b={torch_b.item():.4f}")
```

The key insight (see `[[adam-optimization]]`): without bias correction, the first few steps are dominated by the zero initialization of m and v, making estimates inaccurate. Correcting by 1/(1 − β^t) rescales the moments to their true expected value for the current step count.

---

## Solution 3 — Learning-rate schedule

```python
import math
import torch
from gd.models import LinearModel
from gd.optimizers import SGD

torch.manual_seed(42)
X = torch.randn(100, 3)
y = X @ torch.tensor([1.5, -2.0, 0.7]) + 0.3 + 0.1 * torch.randn(100)

STEPS = 200
LR_MAX = 0.1
LR_MIN = 1e-4


def cosine_lr(t: int, T: int) -> float:
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t / T))


def run(use_schedule: bool) -> list[float]:
    model = LinearModel(w=torch.zeros(3), b=torch.tensor(0.0))
    losses = []
    for t in range(STEPS):
        lr = cosine_lr(t, STEPS) if use_schedule else LR_MAX
        opt = SGD(lr=lr)
        gw, gb = model.gradients(X, y)
        new_w, new_b = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=new_w, b=new_b)
        losses.append(model.loss(X, y).item())
    return losses


constant_losses = run(use_schedule=False)
cosine_losses = run(use_schedule=True)

print(f"Constant LR — final loss: {constant_losses[-1]:.6f}")
print(f"Cosine schedule — final loss: {cosine_losses[-1]:.6f}")
```

**What to observe:** The constant-rate run converges quickly initially but oscillates around the minimum because the step size is always too large for fine-grained refinement. The cosine schedule allows the same aggressive early convergence and then gracefully reduces the step size, enabling the optimizer to settle more precisely. Late-stage loss under cosine annealing is typically 10–100x lower than constant-rate SGD at the same total step count.

---

## Solution 4 — Diverging learning rate

```python
import math
import torch
from gd.models import LinearModel
from gd.optimizers import SGD

torch.manual_seed(42)
X = torch.randn(100, 3)
y = X @ torch.tensor([1.5, -2.0, 0.7]) + 0.3 + 0.1 * torch.randn(100)

# The MSE gradient is (2/n) Xᵀ(Xw+b-y).
# Its Lipschitz constant L ≈ 2 * λ_max(XᵀX) / n.
# SGD converges iff lr < 2/L.  We deliberately exceed this.
BAD_LR = 5.0

model = LinearModel(w=torch.zeros(3), b=torch.tensor(0.0))
opt = SGD(lr=BAD_LR)

for step in range(20):
    loss_val = model.loss(X, y).item()
    print(f"Step {step:2d}: loss = {loss_val:.4f}")
    if not math.isfinite(loss_val):
        print("  --> diverged (nan/inf)")
        break
    gw, gb = model.gradients(X, y)
    new_w, new_b = opt.step([model.w, model.b], [gw, gb])
    model = LinearModel(w=new_w, b=new_b)
```

**Why this diverges (explanation):**

The MSE gradient is (2/n)·XᵀXw + constant terms in w. This is a linear function of w, so the gradient is Lipschitz with constant L = (2/n)·λ_max(XᵀX), where λ_max is the largest eigenvalue of XᵀX. For a step opposite the gradient to decrease the loss, the learning rate must satisfy lr < 2/L. With lr = 5.0 and typical random X, the product lr·L far exceeds 2, so each step overshoots the minimum, landing on a higher part of the loss surface, which produces an even larger gradient on the next step — a positive feedback loop.

**Diagnosing and remedying divergence in practice:**

1. Monitor the gradient norm and the training loss together. Divergence shows up as gradient norm spiking followed by loss exploding, or loss immediately increasing from step 1.
2. Immediate remedies: (a) Reduce learning rate by 10× and restart; (b) apply gradient clipping (e.g., `torch.nn.utils.clip_grad_norm_` with max norm 1.0) to cap the maximum step size regardless of learning rate.

---

## Solution 5 — Vectorized batch loop and benchmark

```python
import torch
import time
from gd.models import LinearModel
from gd.optimizers import SGD

torch.manual_seed(0)
X = torch.randn(1000, 10)
y = X @ torch.ones(10) + torch.randn(1000) * 0.1
EPOCHS = 50

# ── Naive per-sample loop ─────────────────────────────────────────────────────
model_naive = LinearModel(w=torch.zeros(10), b=torch.tensor(0.0))
opt_naive = SGD(lr=0.01)

start = time.perf_counter()
for _ in range(EPOCHS):
    for i in range(len(X)):
        xi = X[i:i+1]
        yi = y[i:i+1]
        gw, gb = model_naive.gradients(xi, yi)
        new_w, new_b = opt_naive.step([model_naive.w, model_naive.b], [gw, gb])
        model_naive = LinearModel(w=new_w, b=new_b)
naive_time = time.perf_counter() - start

# ── Vectorized batch loop ─────────────────────────────────────────────────────
model_batch = LinearModel(w=torch.zeros(10), b=torch.tensor(0.0))
opt_batch = SGD(lr=0.01)

start = time.perf_counter()
for _ in range(EPOCHS):
    gw, gb = model_batch.gradients(X, y)   # one call over all 1000 samples
    new_w, new_b = opt_batch.step([model_batch.w, model_batch.b], [gw, gb])
    model_batch = LinearModel(w=new_w, b=new_b)
batch_time = time.perf_counter() - start

print(f"Naive:      {naive_time:.3f}s  loss={model_naive.loss(X, y):.6f}")
print(f"Vectorized: {batch_time:.3f}s  loss={model_batch.loss(X, y):.6f}")
print(f"Speedup: {naive_time / batch_time:.1f}x")
```

**Why vectorization is faster:**

The per-sample loop performs 1000 separate Python function calls per epoch, each incurring Python interpreter overhead, a separate PyTorch kernel dispatch, and GPU/CPU synchronization. The vectorized version dispatches a single BLAS/cuBLAS GEMM call for the entire 1000×10 matrix multiplication, which saturates hardware parallelism — modern CPUs and GPUs can execute hundreds of multiply-accumulate operations simultaneously. Even though the total FLOP count is identical, the ratio of compute to overhead is orders of magnitude better in the batch case. A typical speedup is 20–200× depending on hardware; this exercise is a direct demonstration of why vectorization is non-negotiable in production ML code.
