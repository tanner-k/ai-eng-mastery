# Vectors and Norms — Interview Prep

## Q&A

1. **Q: What is a vector norm?**
   **A:** A norm is a function that measures vector size and satisfies non-negativity, absolute homogeneity, and the triangle inequality. Common examples are L1, L2, and L-infinity norms.

2. **Q: How do L1 and L2 norms differ in model behavior?**
   **A:** L1 measures total absolute value and tends to encourage sparse solutions when used as a penalty. L2 measures Euclidean length and encourages smooth shrinkage across many coordinates.

3. **Q: What does cosine similarity measure?**
   **A:** It measures directional alignment: `(x dot y) / (||x|| ||y||)`. It removes vector magnitude and keeps the angle information.

4. **Q: When is dot product preferable to cosine similarity?**
   **A:** Dot product is useful when magnitude carries meaningful signal, such as confidence or popularity encoded in embedding norm. Cosine is better when magnitude mostly reflects nuisance factors.

5. **Q: Why is squared L2 norm common in losses?**
   **A:** It is smooth, easy to differentiate, and has gradient x for `(1/2)||x||^2`. The unsquared L2 norm has a division by `||x||` and is not differentiable at zero.

6. **Q: What is gradient norm clipping?**
   **A:** It rescales the full gradient vector when its norm exceeds a threshold. This limits the update size while preserving the direction of the gradient.

7. **Q: What is the relationship between Euclidean distance and cosine similarity for normalized vectors?**
   **A:** If x and y are unit vectors, `||x - y||_2^2 = 2 - 2 cos(theta)`. Ranking by smallest Euclidean distance is equivalent to ranking by largest cosine similarity.

8. **Q: Why can high-dimensional vector intuition be misleading?**
   **A:** Distances and angles concentrate in high dimensions. Random vectors are often nearly orthogonal, and nearest-neighbor gaps can be small, so empirical evaluation matters.

9. **Q: How do norms appear in regularization?**
   **A:** L2 regularization penalizes large weights smoothly, while L1 regularization can drive weights exactly to zero. Both modify the loss landscape and gradient updates.

10. **Q: What numerical issue occurs in cosine similarity near zero vectors?**
    **A:** Division by a tiny norm can produce unstable or undefined values. Implementations usually add epsilon to the denominator and may handle zero vectors explicitly.

## Explain it like a principal

Vector geometry is the language of representation quality and training stability. You should be able to connect an observed metric to a mechanism: rising gradient norms imply unstable updates, embedding norm drift may change retrieval ranking, and regularization strength changes the feasible geometry of model parameters. The senior move is not memorizing formulas; it is choosing the metric whose invariances match the product and training problem.

## Gotchas & follow-ups

- **"Cosine is always better for embeddings."** Not always. If norm contains confidence or relevance information, cosine can throw away useful signal.
- **Ignoring zero-vector behavior.** Cosine similarity and L2 normalization need explicit epsilon handling.
- **Using per-element clipping blindly.** It can change gradient direction; global norm clipping is usually the better default.
- **Conflating L2 norm and squared L2 norm.** They have different gradients and different behavior at zero.
- **Follow-up prompt:** Prove why cosine ranking equals Euclidean ranking after all vectors are L2-normalized.
