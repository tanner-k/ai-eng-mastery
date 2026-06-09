# Gradient Descent — Interview Prep

## Q&A

1. **Q: Why does gradient descent converge to a minimum on convex loss surfaces?**
   **A:** On a strictly convex loss, there is exactly one global minimum and the loss surface has no flat regions or saddle points. The gradient points away from that minimum everywhere, so repeatedly stepping opposite the gradient — with a sufficiently small learning rate — must monotonically decrease the loss. The Lipschitz continuity of the gradient guarantees that the decrease per step is bounded below, so the sequence converges.

2. **Q: What are the tradeoffs between batch GD, SGD, and mini-batch GD?**
   **A:** Batch GD computes the exact gradient over all n examples, which is low-variance but expensive per step and cannot fit large datasets in memory. True SGD uses a single random example per step, giving a high-variance (noisy) gradient estimate that is cheap to compute but converges slowly. Mini-batch GD averages gradients over k ≪ n examples, balancing statistical efficiency with hardware throughput — GPUs can vectorize a batch of 32–512 samples almost as cheaply as a single sample, making this the practical default in deep learning.

3. **Q: What happens when the learning rate is too large or too small?**
   **A:** A learning rate that is too large causes the optimizer to overshoot the minimum on each step; if it exceeds 2/L (where L is the Lipschitz constant of the gradient), training can diverge. A learning rate that is too small makes training painfully slow and can cause the optimizer to stall in flat regions or plateau for many epochs before converging. The optimal learning rate is typically found through a learning-rate range test or a grid search on a log scale.

4. **Q: Describe common learning-rate schedules and when you would use each.**
   **A:** Step decay reduces the learning rate by a constant factor every N epochs, which works well when training loss plateaus predictably. Cosine annealing smoothly decays the rate following a half-cosine curve, often paired with warm restarts (SGDR), and tends to find sharper but often better minima in deep networks. Linear warmup gradually increases the learning rate from zero over the first few hundred steps; this prevents the optimizer from making large, inaccurate updates before the running moments (e.g., in Adam) are properly initialized, and is nearly universal in transformer training.

5. **Q: What is momentum, and why does it help?**
   **A:** Momentum maintains an exponential moving average of past gradients (the velocity vector v) and moves parameters along v rather than the raw gradient. This does two things: it dampens oscillations in directions of high curvature (where the gradient sign flips each step) and it accelerates movement along consistently downhill directions. With β ≈ 0.9, the optimizer effectively integrates the last ~10 gradient steps, smoothing out mini-batch noise and speeding convergence through elongated ravines.

6. **Q: How does RMSProp achieve per-coordinate adaptivity?**
   **A:** RMSProp maintains an exponential moving average of squared gradients s per parameter. The effective learning rate for each coordinate is η / √(s + ε), so coordinates that historically receive large gradients get their step size shrunk while coordinates with small historical gradients are amplified. This normalizes the scale of each parameter direction, making the method much more robust to poorly conditioned problems (elongated loss ellipses) without requiring the user to tune separate rates per parameter.

7. **Q: How does Adam fuse momentum and RMSProp, and why does bias correction matter?**
   **A:** Adam tracks a first moment m (momentum, EMA of gradients) and a second moment v (RMSProp, EMA of squared gradients). Without correction both m and v are initialized to zero, so early estimates are strongly biased toward zero — step sizes would be too small for the first several hundred updates. Bias correction divides each moment by (1 − β^t), inflating the estimates back to the true scale. By step t ≈ 1000 with β₂ = 0.999, the correction factor has decayed to roughly one and its effect disappears.

8. **Q: Are saddle points or local minima the bigger problem in deep learning?**
   **A:** For modern over-parameterized networks, saddle points are empirically the more common obstruction. In high-dimensional spaces most critical points have at least one direction of negative curvature (making them saddle points, not minima), and the local minima that do exist tend to have similar loss values to the global minimum. SGD's gradient noise and momentum help escape saddle points because the stochastic perturbation is unlikely to be aligned with the flat saddle direction. Local minima that are actually harmful are rare in practice but can appear in small, under-parameterized models.

9. **Q: How does gradient descent relate to vanishing and exploding gradients?**
   **A:** In deep networks, gradients are backpropagated through many layers via the chain rule, which multiplies Jacobians together. If the singular values of those Jacobians are consistently less than one, the gradient signal shrinks exponentially as it travels toward earlier layers (vanishing), effectively preventing those layers from learning. If the singular values are consistently greater than one, the gradient explodes, causing numerical overflow and training instability. Both pathologies are GD failures at the heart — the algorithm has no useful signal to follow.

10. **Q: How does batch size affect training dynamics?**
    **A:** Larger batches produce lower-variance gradient estimates, enabling larger learning rates and more stable steps, but generalization often worsens — large-batch training tends to converge to sharp minima with poor test performance (the "generalization gap"). Smaller batches introduce gradient noise that acts as an implicit regularizer and helps navigate the loss landscape more broadly. There is also a practical ceiling: beyond a critical batch size you gain diminishing returns in convergence per compute unit, so linear scaling of learning rate with batch size (the linear scaling rule) breaks down.

11. **Q: When does gradient descent diverge or completely fail?**
    **A:** GD diverges when the learning rate exceeds 2/L for convex losses (the Lipschitz constant), causing loss to increase on every step. It can also stall in flat regions (plateaus, saddle points) where the gradient is near zero. On non-smooth losses (L1, hinge) GD is not defined at kinks; subgradient methods must be used instead. Gradient explosion in deep networks is a practical failure mode; gradient clipping is the standard fix. Finally, if the data distribution shifts mid-training (covariate shift without re-normalization), GD may converge to the wrong solution.

12. **Q: What distinguishes convex from non-convex optimization, and why does it matter?**
    **A:** A function is convex if every chord between two points on its graph lies above (or on) the graph, implying any local minimum is also a global minimum. Convex losses guarantee that GD will find the optimal solution given a proper learning rate. Deep neural networks are non-convex: multiple local minima, saddle points, and flat regions exist. In practice, over-parameterization, SGD noise, and careful initialization make non-convex training work reliably, but there are no theoretical guarantees about finding a global minimum.

13. **Q: How do first-order and second-order optimization methods differ?**
    **A:** First-order methods (GD, SGD, Adam) use only gradient information — the first derivative of the loss. Second-order methods (Newton's method, natural gradient, L-BFGS) also use curvature information from the Hessian (or its approximation), enabling larger, more accurate steps. Second-order methods converge in far fewer steps on convex problems but computing or inverting the full Hessian is O(p²) in memory and O(p³) in compute (where p is the number of parameters), making them infeasible for large neural networks. Adam can be seen as a diagonal approximation to natural gradient descent.

14. **Q: What is gradient clipping, and when should you apply it?**
    **A:** Gradient clipping rescales the gradient vector (or clips each element independently) whenever its norm exceeds a threshold, preventing single large gradient steps from destabilizing training. It is standard practice for RNNs and Transformers, where long backpropagation paths through time make gradient explosion common. Global norm clipping (rescaling so ‖g‖ ≤ τ) is preferred over per-element clipping because it preserves the direction of the gradient. A typical threshold is τ = 1.0; the right value is identified when the "grad norm" training metric regularly exceeds it before clipping.

15. **Q: What is the difference between weight decay and L2 regularization in the context of GD optimizers?**
    **A:** In vanilla SGD they are mathematically equivalent: L2 regularization adds λ‖w‖² to the loss, producing a gradient term that is subtracted proportionally to w, which is identical to decaying w by (1 − η·λ) each step. However, in adaptive methods like Adam, L2 regularization interacts with the second-moment normalization — the penalty gradient is scaled down by the adaptive term — so the effective regularization strength depends on gradient history and varies across parameters. True weight decay (AdamW) applies the decay directly to the parameter update, decoupled from the gradient scaling, giving uniform and predictable regularization regardless of gradient magnitude.

16. **Q: Why is learning-rate warmup important for adaptive optimizers like Adam?**
    **A:** In the first few steps, Adam's second-moment estimate v is close to zero (initialized to zero, slowly accumulating). Dividing by √v produces a very large effective learning rate that can make the optimizer take enormous, destabilizing steps in arbitrary directions before it has a useful estimate of curvature. Warmup starts the learning rate at a tiny value and linearly increases it to the target rate over the first several hundred to few thousand steps, allowing the moment estimates to stabilize before large steps are taken. This is especially critical in transformer training, where the loss landscape at initialization is poorly conditioned.

---

## Explain it like a principal

Gradient descent is the optimization primitive on which all modern ML training rests. At senior/architectural level, the key decisions are not whether to use it but which variant, at what scale, and with what schedule. Batch size, learning rate, and optimizer choice form a coupled system: larger batches let you use larger learning rates, but you trade off implicit regularization and memory budget. Adam is the default because it adapts to gradient magnitude heterogeneity across layers and is robust across architectures, but it can generalize worse than SGD with momentum on well-tuned vision tasks. The real engineering challenge is that the loss landscape of a large model is never directly observable — you navigate it through proxies (training loss curve, gradient norms, learning-rate sensitivities) and must diagnose divergence, underfitting, and over-smoothing from telemetry alone. Knowing how each optimizer behaves mechanistically (what its moments track, how bias correction damps early steps, when adaptive methods suppress useful gradient signal) is what separates engineers who tune hyperparameters empirically from those who can reason about training dynamics from first principles.

---

## Gotchas & follow-ups

- **"Adam always outperforms SGD."** False. On image classification with careful tuning, SGD + momentum + cosine schedule frequently beats Adam in final accuracy. Adam reaches low training loss faster but can converge to sharper minima. When pushed on optimizer choice, explain the generalization gap trade-off, not just speed.

- **"Momentum accumulates past gradients."** Common imprecision. Momentum accumulates an exponential moving average of past gradients — it does not sum them without decay. This means the contribution of step t−k decays as βᵏ, so only recent history matters. Interviewers who ask "how many past gradients does momentum remember?" expect you to explain the EMA, not give a fixed number.

- **Confusing L2 regularization and weight decay for Adam.** The distinction matters in production: using `weight_decay` in `torch.optim.Adam` (before PyTorch merged AdamW behavior) applied the penalty through the gradient, weakening it via adaptive scaling. AdamW decouples them. Know which you are actually using, and note that the original Adam paper did not include weight decay correctly.

- **Saddle points versus flat regions.** When the gradient is near zero, you cannot immediately distinguish a saddle point (gradient is exactly zero but curvature is mixed) from a wide flat basin (gradient is near zero over a large region). Follow-up: how would you detect a saddle versus a minimum in practice? (Inspect eigenvalues of the Hessian or monitor gradient norm vs. loss stagnation.)

- **"Larger batch size means faster training."** Only in wall-clock time if you can fill the GPU. Beyond a critical batch size, convergence per sample degrades and you need more total gradient steps to reach the same loss, negating the throughput gain. Expect the follow-up: "What is the linear scaling rule, and when does it break down?"

- **Warmup and the first few steps of Adam.** Many candidates can explain Adam's update formula but cannot explain why warmup is necessary. Be ready to walk through what happens to the bias-corrected estimates at t=1 with β₂=0.999: v̂ = v/(1-0.999¹) = v/0.001, so an uncorrected v near zero gives √v̂ → 0 and an astronomically large effective step. Warmup prevents this by keeping η small until the estimates stabilize.
