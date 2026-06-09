# kNN and Naive Bayes — Interview Prep

## Q&A

1. **Q: What is kNN's training phase?**
   **A:** Mostly storing the training data, aside from optional indexing or preprocessing.

2. **Q: Why is kNN prediction expensive?**
   **A:** A query may need distances to many or all training examples.

3. **Q: Why does feature scaling matter for kNN?**
   **A:** Distance metrics are scale-sensitive; large-scale features dominate.

4. **Q: How does `k` affect bias and variance?**
   **A:** Small `k` has low bias and high variance. Large `k` smooths predictions, increasing bias and reducing variance.

5. **Q: What is Naive Bayes' independence assumption?**
   **A:** Features are conditionally independent given the class.

6. **Q: Why can Naive Bayes work even when independence is false?**
   **A:** It can still rank classes correctly if the accumulated evidence points in the right direction.

7. **Q: What is Laplace smoothing?**
   **A:** Adding pseudo-counts to avoid zero probabilities for unseen tokens or feature values.

8. **Q: Why use log probabilities?**
   **A:** Products of many small probabilities underflow; logs turn products into sums.

9. **Q: Gaussian versus multinomial Naive Bayes?**
   **A:** Gaussian models continuous features with class-specific normal distributions. Multinomial models count data such as bag-of-words text.

## Explain it like a principal

kNN and Naive Bayes are baseline opposites. kNN trusts local geometry and postpones work until prediction time. Naive Bayes commits to a simple generative model and predicts extremely fast. The engineering choice depends on feature space, latency, data size, interpretability, and whether distance or conditional evidence is the right inductive bias.

## Gotchas & follow-ups

- **"kNN has no hyperparameters."** `k`, distance metric, weighting, scaling, and indexing all matter.
- **"Naive Bayes requires independence to be true."** It requires independence for calibrated likelihoods, but classification can still perform well when the assumption is approximate or useful.
- **"Smoothing is optional."** Without smoothing, unseen tokens can zero out an entire class score.
- **Follow-up:** How would you make kNN work for millions of embeddings with strict latency requirements?
