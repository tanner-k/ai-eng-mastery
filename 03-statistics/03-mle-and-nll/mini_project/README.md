# MLE and Negative Log-Likelihood — Mini-Project: Loss From Likelihood

Build a small comparison that derives common losses from probability models.

---

## Goal

Show that MSE, binary cross-entropy, and multiclass cross-entropy are negative log-likelihoods under different target distributions.

---

## Data Setup

Generate synthetic data inside the future script:

- Gaussian regression targets with known residual variance.
- Bernoulli binary labels with known probabilities.
- Categorical labels from three-class probability vectors.

No external dataset is required.

---

## Implementation Tasks

1. Create a future script such as `mini_project/loss_from_likelihood.py`.
2. Implement Gaussian NLL for fixed variance.
3. Implement Bernoulli NLL from probabilities.
4. Implement categorical NLL from class probabilities.
5. Compare each manual value with the corresponding PyTorch loss function when available.
6. Demonstrate why direct `log(probability)` is unstable near `0`.
7. Print a table of manual and library-computed losses.

---

## Expected Workflow

After creating the script, run it from this topic directory with a command like:

```bash
uv run python mini_project/loss_from_likelihood.py
```

The script should generate all examples in memory.

---

## Expected Outputs

- Matching manual and framework loss values for simple examples.
- A clear note that fixed-variance Gaussian NLL and MSE have the same minimizer.
- A numerical-stability demonstration for probabilities close to zero.

---

## Writeup Prompt

Write 6-8 sentences explaining why likelihood assumptions matter when choosing a loss. Include one example where changing the assumed noise model would change the training objective.

---

## Optional Extensions

- Add weighted NLL for imbalanced classification.
- Add heteroskedastic Gaussian NLL where the model predicts variance.
- Compare NLL with accuracy on deliberately miscalibrated probabilities.
