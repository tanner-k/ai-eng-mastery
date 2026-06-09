# K-Means — Interview Prep

## Q&A

1. **Q: What objective does k-means minimize?**
   **A:** Sum of squared distances from each point to its assigned centroid.

2. **Q: What are the two k-means steps?**
   **A:** Assign points to nearest centroids, then update centroids to assigned-point means.

3. **Q: Does k-means find the global optimum?**
   **A:** Not generally. It converges to a local optimum depending on initialization.

4. **Q: Why use k-means++?**
   **A:** It spreads initial centroids apart, improving convergence and solution quality.

5. **Q: Why standardize features?**
   **A:** Euclidean distance is scale-sensitive; large-scale features dominate assignments.

6. **Q: How do you choose K?**
   **A:** Combine domain knowledge, elbow/silhouette analysis, stability, and downstream performance.

7. **Q: What cluster shapes does k-means prefer?**
   **A:** Roughly spherical, similar-size clusters under Euclidean distance.

8. **Q: How do outliers affect k-means?**
   **A:** They can pull centroids away from dense regions because means are not robust.

9. **Q: What is inertia?**
   **A:** The within-cluster sum of squared distances to centroids.

## Explain it like a principal

K-means is a fast prototype learner, not a truth machine. Its clusters reflect the chosen feature space, scaling, distance metric, and value of `K`. Used well, it is a useful compression and segmentation tool; used carelessly, it gives arbitrary group labels with unwarranted meaning. Strong engineering use validates cluster stability and downstream usefulness.

## Gotchas & follow-ups

- **"K-means is deterministic."** Only after initialization; random seeds can produce different solutions.
- **"Lower inertia always means better clustering."** Inertia always decreases as `K` increases, even if clusters are not meaningful.
- **"Cluster IDs are ordered."** Labels are arbitrary; cluster 0 has no inherent meaning.
- **Follow-up:** How would you cluster embeddings where cosine similarity matters more than Euclidean distance?
