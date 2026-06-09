# Gradient Descent

## Overview

Gradient descent is the workhorse optimization algorithm behind virtually every machine learning and deep learning model trained today. The core idea is simple: given a differentiable loss function L(θ) that measures how wrong our model is, we can iteratively nudge the parameters θ in the direction that reduces the loss most quickly. That direction is the negative gradient −∇L, which points "downhill" on the loss surface.

Neural networks can have billions of parameters and are trained on billions of examples. Gradient descent — and its stochastic and adaptive variants — scales to this regime because computing a gradient is a single backward pass through the computation graph, not a search over the parameter space. Without gradient descent (or a variant of it), modern deep learning would not be computationally feasible.

## Math / Derivation

### MSE Loss and Its Gradients

For linear regression with prediction ŷ = Xw + b, the mean-squared error loss is:

```
L(w, b) = (1/n) · Σᵢ (Xᵢw + b − yᵢ)²
```

Differentiating with respect to w and b gives:

```
∇_w L = (2/n) · Xᵀ(Xw + b − y)
∇_b L = (2/n) · Σ(Xw + b − y)
```

These are the analytic gradients implemented in `gd/models.py` and validated against `torch.autograd` in the notebook.

### The Update Rule

At each step t, parameters are updated by stepping opposite the gradient, scaled by the learning rate η:

```
θ ← θ − η · ∇L(θ)
```

Repeating this until convergence (or a fixed number of steps) is gradient descent.

### Batch vs. SGD vs. Mini-batch

| Variant | Data per step | Notes |
|---|---|---|
| Batch GD | All n examples | Exact gradient; expensive per step |
| Stochastic GD (SGD) | 1 example | Noisy but fast; can escape shallow minima |
| Mini-batch GD | k examples (k ≪ n) | Best of both: vectorization efficiency + gradient noise |

In practice, "SGD" in deep learning frameworks means mini-batch SGD. The noise introduced by sampling acts as a regularizer and helps traverse flat regions and avoid sharp minima.

### Momentum

Classical SGD is prone to oscillation in directions with high curvature. Momentum maintains an exponential moving average of past gradients (velocity v) and uses it to smooth and accelerate the update:

```
v ← β·v + (1 − β)·∇L
θ ← θ − η·v
```

With β ≈ 0.9, the optimizer accumulates speed in consistent directions while dampening oscillation across ravines.

### RMSProp

RMSProp addresses the problem that a single global learning rate is inappropriate when different parameters have very different gradient magnitudes. It maintains a per-coordinate exponential moving average of squared gradients s:

```
s ← β·s + (1 − β)·(∇L)²
θ ← θ − η · ∇L / (√s + ε)
```

Coordinates with large historical gradients get a smaller effective learning rate; coordinates with small gradients get a larger effective rate. ε (typically 1e-8) prevents division by zero.

### Adam

Adam (Adaptive Moment Estimation) combines Momentum and RMSProp, tracking both a first moment m and a second moment v of the gradients. Crucially, it applies bias correction to both moments to compensate for their zero initialization at the start of training:

```
m ← β₁·m + (1 − β₁)·∇L          # first moment (mean)
v ← β₂·v + (1 − β₂)·(∇L)²       # second moment (uncentered variance)

m̂ = m / (1 − β₁ᵗ)               # bias-corrected first moment
v̂ = v / (1 − β₂ᵗ)               # bias-corrected second moment

θ ← θ − η · m̂ / (√v̂ + ε)
```

With default hyperparameters β₁ = 0.9, β₂ = 0.999, ε = 1e-8, Adam converges quickly across a wide range of problems and is the default choice for most deep learning workloads.

## Intuition

Think of the loss surface as a hilly landscape. The model's current parameters are a ball sitting somewhere on that landscape. The gradient tells us the slope at the ball's exact position — the direction of steepest ascent. We roll the ball in the opposite direction (steepest descent) by a distance proportional to the learning rate.

**Learning rate** is the step size. Too large and the ball overshoots the valley and may diverge; too small and training takes forever. A well-tuned learning rate is often the single most impactful hyperparameter.

**Momentum** is physical inertia: the ball remembers which way it was rolling and tends to keep going. This smooths out oscillations in narrow ravines and lets the optimizer build up speed on long, consistent downward slopes.

**RMSProp** scales the learning rate differently for each axis of the parameter space. Steep axes (large gradient history) get smaller steps; shallow axes get larger steps. The ball navigates elongated elliptical valleys much more efficiently.

**Adam** gets both benefits simultaneously: momentum for direction smoothing and per-coordinate scaling for step size. The bias correction ensures that the very first steps are not dominated by the zero initialization of the running averages.

## When & Why

**Convex surfaces** (like MSE on linear regression) have a single global minimum; any variant of gradient descent will find it given a suitable learning rate. This makes the 2-parameter linear regression in `implementation.ipynb` an ideal teaching example — you can plot the entire loss surface and watch the optimizer walk down it.

**Non-convex surfaces** (deep neural networks) have many local minima and saddle points. Saddle points are problematic for pure gradient descent because the gradient is zero there, but they are typically surrounded by directions with negative curvature that SGD noise or momentum can escape. Empirically, the local minima found by SGD in practice have similar loss to the global minimum for over-parameterized networks.

**Conditioning** refers to the ratio of the largest to smallest eigenvalues of the Hessian. Poorly conditioned problems (elongated valleys) cause vanilla SGD to oscillate badly. Adaptive methods like RMSProp and Adam precondition the gradient, effectively normalizing the scale of each coordinate, which makes them far more robust to poor conditioning.

## Implementation

The from-scratch implementation lives in `gd/`:

- `gd/models.py` — `LinearModel`: immutable dataclass with analytic `gradients()` and `loss()` methods.
- `gd/optimizers.py` — `SGD`, `Momentum`, `RMSProp`, `Adam`: stateful optimizers whose `step(params, grads)` method returns **new** parameter tensors (immutable parameter pattern; the optimizers themselves maintain internal moment state).

The executable notebook `implementation.ipynb` walks through:

1. Generating a synthetic 1-feature linear regression dataset.
2. Plotting the 2-D MSE loss surface over (w, b).
3. Running from-scratch SGD and visualizing the descent path on the contour plot.
4. Validating hand-derived gradients against `torch.autograd` (exact match to 1e-5).
5. Comparing convergence curves for SGD, Momentum, RMSProp, and Adam.
6. Reproducing the Adam result with `torch.optim.Adam` and asserting the two match to 1e-2.

## Cross-links

- `[[chain-rule]]` — gradients are computed via the chain rule applied to the computational graph.
- `[[gradients-and-jacobians]]` — the vector calculus behind ∇L.
- `[[loss-functions]]` — MSE is one of many choices; the gradient form changes with the loss.
- `[[l1-l2-regularization]]` — adds a penalty term to L, changing the gradient but not the update rule structure.
- `[[adam-optimization]]` — deep dive into Adam's convergence properties and variants (AdamW, Nadam).

## Resources

- Sebastian Ruder, "An overview of gradient descent optimization algorithms." arXiv:1609.04747 (2016). <https://arxiv.org/abs/1609.04747>
- Diederik P. Kingma & Jimmy Ba, "Adam: A Method for Stochastic Optimization." ICLR 2015. arXiv:1412.6980. <https://arxiv.org/abs/1412.6980>
