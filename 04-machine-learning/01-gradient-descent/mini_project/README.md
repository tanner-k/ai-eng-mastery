# Gradient Descent — Mini-Project: Optimizer Bake-Off

Train four optimizers (SGD, Momentum, RMSProp, Adam) on the same real regression dataset, plot their convergence curves, and write a brief analysis of which optimizer wins and why.

---

## Goal

Compare the practical convergence behavior of the four optimizers implemented in `gd/optimizers.py` on a real (small) dataset under identical conditions. The deliverable is a short Python script plus a written summary.

---

## Dataset

Use `sklearn.datasets.load_diabetes` — a 442-sample, 10-feature standardized regression dataset predicting disease progression. It is small enough to run in seconds on CPU and large enough to show meaningful convergence differences.

Add `scikit-learn` to the project dependencies:

```bash
uv add scikit-learn
```

Alternatively, if you prefer zero extra dependencies, generate a small synthetic CSV:

```python
import torch, csv

torch.manual_seed(0)
n, d = 500, 8
X = torch.randn(n, d)
w_true = torch.randn(d)
y = X @ w_true + 0.5 * torch.randn(n)

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"x{i}" for i in range(d)] + ["y"])
    for xi, yi in zip(X.tolist(), y.tolist()):
        writer.writerow(xi + [yi])
```

---

## Setup

```bash
cd 04-machine-learning/01-gradient-descent
uv sync          # install project dependencies
```

---

## Implementation Spec

Create `mini_project/bakeoff.py` with the following structure:

### 1. Load and standardize data

```python
from sklearn.datasets import load_diabetes
import torch

data = load_diabetes()
X_np, y_np = data.data, data.target

# Standardize features to zero mean, unit variance
X_mean = X_np.mean(axis=0)
X_std  = X_np.std(axis=0) + 1e-8
X_np   = (X_np - X_mean) / X_std

# Standardize target
y_mean = y_np.mean()
y_std  = y_np.std()
y_np   = (y_np - y_mean) / y_std

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)
```

Standardization is mandatory. Without it, features on different scales make adaptive vs. non-adaptive optimizer differences impossible to interpret cleanly.

### 2. Train each optimizer

Run each optimizer for 500 steps (full-batch GD — all 442 samples per step). Record the training loss at every step.

```python
from gd.models import LinearModel
from gd.optimizers import SGD, Momentum, RMSProp, Adam

configs = {
    "SGD":      SGD(lr=0.05),
    "Momentum": Momentum(lr=0.05, beta=0.9),
    "RMSProp":  RMSProp(lr=0.01, beta=0.99),
    "Adam":     Adam(lr=0.01, beta1=0.9, beta2=0.999),
}

STEPS = 500
results: dict[str, list[float]] = {}

for name, opt in configs.items():
    model = LinearModel(
        w=torch.zeros(X.shape[1]),
        b=torch.tensor(0.0),
    )
    losses = []
    for _ in range(STEPS):
        losses.append(model.loss(X, y).item())
        gw, gb = model.gradients(X, y)
        new_w, new_b = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=new_w, b=new_b)
    results[name] = losses
```

### 3. Plot convergence

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
for name, losses in results.items():
    ax.plot(losses, label=name)

ax.set_xlabel("Step")
ax.set_ylabel("MSE Loss (standardized)")
ax.set_title("Optimizer Bake-Off — Diabetes Dataset")
ax.legend()
ax.set_yscale("log")         # log scale reveals early vs. late-stage behavior
fig.tight_layout()
fig.savefig("mini_project/convergence.png", dpi=150)
print("Saved convergence.png")
```

### 4. Print summary table

```python
print(f"\n{'Optimizer':<12} {'Initial Loss':>14} {'Final Loss':>14} {'Reduction':>12}")
print("-" * 56)
initial = results["SGD"][0]
for name, losses in results.items():
    reduction = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"{name:<12} {losses[0]:>14.4f} {losses[-1]:>14.4f} {reduction:>11.1f}%")
```

---

## Commands to Run

```bash
# From the gradient-descent directory:
uv run python mini_project/bakeoff.py
```

To run with the synthetic CSV alternative (no sklearn):

```bash
uv run python mini_project/generate_data.py   # creates data.csv
uv run python mini_project/bakeoff.py --csv data.csv
```

---

## What to Look For

**Convergence speed (early steps 0–50):** Adaptive methods (RMSProp, Adam) typically drop loss much faster in the first 50 steps because they compensate for feature scale differences that remain even after standardization. On a well-standardized dataset the gap narrows, but Adam's momentum term often still gives it a head start.

**Convergence quality (steps 100–500):** Compare final losses on a log scale. Adam and RMSProp usually achieve a lower floor faster. SGD without momentum may still be converging slowly at step 500 — this is expected at learning rate 0.05 for a convex problem of this size; the rate is deliberately chosen to make the comparison instructive rather than to maximally tune each optimizer.

**Oscillation:** If any optimizer's loss curve shows sawtooth patterns (loss bouncing up and down rather than monotonically decreasing), the learning rate is too large for that optimizer. Note which methods are sensitive to this.

**Momentum vs. SGD:** Momentum's curve should be smoother and reach the same neighborhood of the minimum in fewer steps. If they converge to the same final loss, momentum simply got there faster — this is its practical value on convex problems.

**Adam's bias correction:** The first few steps of Adam may appear to "waste" a few steps — the loss may decrease more slowly than RMSProp at step 1–5. This is the bias correction paying its tax: Adam is being conservative until its moment estimates stabilize.

---

## Writeup Prompt

After running the script, write a short paragraph (5–8 sentences) answering:

1. Which optimizer reached the lowest loss at step 500?
2. Which reached 50% loss reduction first (scan the loss arrays to find the step)?
3. Did any optimizer diverge or oscillate? What does that suggest about its learning rate?
4. Based on these results, which optimizer would you choose for a production regression pipeline and why? Consider convergence speed, final quality, and sensitivity to hyperparameter tuning.

Save your writeup as `mini_project/analysis.md`.
