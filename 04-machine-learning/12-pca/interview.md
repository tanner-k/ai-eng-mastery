# Principal Component Analysis — Interview Prep

## Q&A

1. **Q: What does PCA find?**
   **A:** Orthogonal directions of maximum variance in centered data.

2. **Q: Why center data before PCA?**
   **A:** PCA should capture variance around the mean, not directions caused by feature offsets.

3. **Q: How is PCA related to eigendecomposition?**
   **A:** Principal components are eigenvectors of the covariance matrix.

4. **Q: How is PCA related to SVD?**
   **A:** Right singular vectors of the centered data matrix are principal directions.

5. **Q: What is explained variance ratio?**
   **A:** The fraction of total variance captured by each component.

6. **Q: Does PCA use labels?**
   **A:** No. It is unsupervised.

7. **Q: Can PCA hurt supervised performance?**
   **A:** Yes. Low-variance directions can be predictive, and PCA may discard them.

8. **Q: What is whitening?**
   **A:** Scaling principal components so they have unit variance.

9. **Q: How do you choose number of components?**
   **A:** Use explained variance, reconstruction error, visualization needs, or downstream validation metrics.

## Explain it like a principal

PCA is a controlled linear compression of feature space. It is powerful when redundancy and correlation are the problem, but it is blind to labels and product outcomes. Mature use treats PCA as a fitted preprocessing transform with leakage controls, validates component count downstream, and remembers that component interpretability is weaker than original-feature interpretability.

## Gotchas & follow-ups

- **"PCA selects the most important features."** It creates new linear combinations; it does not select original columns.
- **"Highest variance means most predictive."** Not necessarily. PCA is unsupervised.
- **"Fit PCA once on all data for stability."** That leaks validation/test distribution into training.
- **Follow-up:** Why might standardization before PCA be necessary, and when might it be inappropriate?
