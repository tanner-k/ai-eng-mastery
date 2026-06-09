# kNN and Naive Bayes

## Overview

k-nearest neighbors (kNN) and Naive Bayes are simple baselines with opposite assumptions. kNN is instance-based: it stores training examples and predicts from nearby points. Naive Bayes is probabilistic: it estimates class-conditional feature likelihoods and applies Bayes' rule with a conditional independence assumption.

Both are useful interview topics because they expose core ideas: distance metrics, feature scaling, nonparametric prediction, Bayes' rule, smoothing, and baseline discipline.

## Math / Derivation

For kNN classification, find the `k` training examples closest to query `x` under a distance such as Euclidean:

```
d(x, x_i) = ||x - x_i||_2
```

Predict by majority vote or distance-weighted vote. For regression, predict the mean or weighted mean of neighbor targets.

Naive Bayes predicts:

```
argmax_y p(y | x) = argmax_y p(y) p(x | y)
```

The naive assumption factorizes features given the class:

```
p(x | y) = product_j p(x_j | y)
```

For multinomial text classification with counts `x_j`, the log score is:

```
log p(y) + sum_j x_j log p(token_j | y)
```

Laplace smoothing estimates token probabilities as:

```
p(token_j | y) = (count_{j,y} + alpha) / (total_count_y + alpha * V)
```

## Intuition

kNN says, "similar examples should have similar labels." It has almost no training cost but expensive prediction and strong dependence on the feature space.

Naive Bayes says, "combine many weak pieces of evidence independently." The independence assumption is often false, but the classifier can still work well, especially for text, because the log evidence accumulates in useful directions.

## When & Why

Use kNN for small datasets, sanity checks, retrieval-like prediction, and embeddings where neighborhood structure is meaningful. Scale features and choose distance carefully. Prediction cost grows with stored data unless approximate nearest-neighbor indexing is used.

Use Naive Bayes for fast text baselines, sparse count features, and low-data classification. It trains quickly, handles high-dimensional sparse data well, and is easy to inspect. It may produce poorly calibrated probabilities when feature independence is badly violated.

## Implementation

A later implementation pass should implement kNN classification/regression with vectorized distance computation and Naive Bayes for categorical or multinomial text-like counts. It should compare feature scaling effects for kNN, smoothing effects for Naive Bayes, and evaluate both on synthetic datasets with appropriate metrics.

The implementation should use log probabilities for Naive Bayes to avoid underflow from multiplying many small probabilities.

## Cross-links

- `[[probability-basics]]` — Naive Bayes applies Bayes' rule.
- `[[gaussian-distribution]]` — Gaussian Naive Bayes models continuous features.
- `[[vectors-and-norms]]` — kNN depends on distance metrics.
- `[[evaluation-metrics]]` — baselines should be judged with task-appropriate metrics.
- `[[pca]]` — dimensionality reduction can change neighborhood structure.

## Resources

- Cover and Hart, "Nearest Neighbor Pattern Classification." 1967.
- Domingos and Pazzani, "On the Optimality of the Simple Bayesian Classifier under Zero-One Loss." 1997.
- Manning, Raghavan, and Schütze, *Introduction to Information Retrieval*, text classification chapters.
