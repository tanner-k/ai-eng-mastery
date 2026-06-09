# Eigendecomposition and SVD — Mini-Project: Low-Rank Signal Finder

Build a synthetic low-rank matrix experiment that uses SVD to recover structure, compress the matrix, and explain reconstruction quality.

## Goal

Create a script that generates a noisy low-rank matrix, computes its singular values, reconstructs rank-k approximations, and reports how much error remains as k increases.

## Dataset

Generate synthetic data inside the script:

```python
import torch

torch.manual_seed(0)
m, n, true_rank = 80, 40, 4
U_true = torch.randn(m, true_rank)
V_true = torch.randn(n, true_rank)
signal = U_true @ V_true.T
noise = 0.1 * torch.randn(m, n)
A = signal + noise
```

No external dataset is required.

## Implementation Tasks

Create a future file such as `mini_project/low_rank_signal.py` and implement:

1. Compute SVD using `torch.linalg.svd`.
2. Print the top singular values and identify the visible spectral drop.
3. Reconstruct rank-k approximations for k in `[1, 2, 4, 8, 16]`.
4. Report relative Frobenius reconstruction error for each k.
5. Verify that `A.T @ A` eigenvalues match squared singular values within tolerance.
6. Add a short PCA-style interpretation: which k would you choose and why?

## Expected Workflow

After creating the script, run it from this topic directory:

```bash
uv run python mini_project/low_rank_signal.py
```

## Expected Outputs

The script should print:

- Matrix shape and true synthetic rank.
- Top singular values.
- A table of k versus relative reconstruction error.
- Max difference between eigenvalues of `A.T @ A` and squared singular values.
- A recommended compression rank with a brief explanation.

The error should drop sharply near the true rank, then improve more slowly as additional components mostly fit noise.

## Writeup Prompt

Write 5-8 sentences answering:

1. Where did the singular value spectrum show an elbow?
2. Which rank gave the best compression-quality tradeoff?
3. Why does truncating after the true rank not remove all error?
4. What downstream checks would be needed before compressing a real embedding or weight matrix?

## Optional Extensions

- Plot singular values on a log scale after creating the script.
- Compare noiseless and noisy matrices.
- Use the same data to run PCA on centered rows and interpret principal directions.
