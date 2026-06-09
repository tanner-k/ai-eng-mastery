# L1 and L2 Regularization — Interview Prep

## Q&A

1. **Q: Why does regularization improve generalization?**
   **A:** It restricts the hypothesis space or penalizes complexity, reducing variance and sensitivity to noise in the training set.

2. **Q: Why does L1 produce sparse coefficients?**
   **A:** The L1 penalty has a non-smooth kink at zero and a constant-magnitude subgradient away from zero. This makes exact zero coefficients optimal for weak features.

3. **Q: Why does L2 rarely produce exact zeros?**
   **A:** Its gradient is proportional to weight size, so the shrinkage becomes smaller as weights approach zero.

4. **Q: What is ridge regression?**
   **A:** Linear regression with an L2 penalty on coefficients.

5. **Q: What is lasso?**
   **A:** Linear regression with an L1 penalty, commonly used for sparse feature selection.

6. **Q: What is Elastic Net?**
   **A:** A combination of L1 and L2 penalties. It encourages sparsity while stabilizing selection among correlated features.

7. **Q: How do you choose `lambda`?**
   **A:** Tune on validation data, usually on a logarithmic scale. The right value depends on data size, noise, feature scale, and model capacity.

8. **Q: Why should features be standardized before L1 or L2?**
   **A:** Penalties act on coefficient magnitudes. Without standardization, feature scale changes the effective amount of regularization per feature.

9. **Q: Is L2 regularization the same as weight decay?**
   **A:** For vanilla SGD, they are equivalent up to convention. For adaptive optimizers, decoupled weight decay is different and usually preferred.

## Explain it like a principal

Regularization is how you encode a prior preference before the validation set tells you what generalizes. L2 says useful models should be smooth and distributed. L1 says useful models should be sparse. In large systems, the practical question is not only validation score; it is stability under retraining, interpretability, feature cost, and how regularization interacts with the optimizer. Strong candidates explain the mechanism and the operational tradeoff.

## Gotchas & follow-ups

- **"L1 is always better because it selects features."** Sparse can be unstable when features are correlated; Elastic Net may be better.
- **"More regularization always improves validation."** Past the optimum, regularization increases bias and hurts both train and validation performance.
- **"Weight decay is just L2."** Ask which optimizer is being used. With Adam, the distinction matters.
- **Follow-up:** Why should bias terms and normalization parameters often be excluded from weight decay?
