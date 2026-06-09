# Principal Component Analysis — Mini-Project: Compression and Leakage

## Goal

Implement PCA and study explained variance, reconstruction error, visualization, and the effect of fitting transforms on the wrong split.

## Dataset

Generate synthetic data from three latent factors mixed into 20 observed features with added noise. Optionally generate labels from one latent factor for downstream evaluation.

## Implementation tasks

Create `mini_project/pca_compression.py` in a future implementation pass. It should:

1. Generate train and validation data.
2. Fit PCA on centered training data using SVD.
3. Transform validation data using the training mean and components.
4. Plot explained variance ratio.
5. Compute reconstruction error for component counts 1 through 10.
6. Compare downstream linear classification or regression with and without PCA.
7. Demonstrate the metric difference when PCA is incorrectly fit on all data.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/pca_compression.py
```

## Expected outputs

- Explained variance table or plot.
- Reconstruction error by component count.
- A 2D PCA visualization.
- A note identifying any leakage effect.

## Writeup prompt

Explain how many components you would keep and why. Separate the unsupervised reconstruction argument from the supervised validation-metric argument.

## Optional extensions

- Add whitening.
- Compare covariance eigendecomposition with SVD.
- Add a low-variance predictive feature and observe PCA failure.
