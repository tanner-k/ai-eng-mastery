# Support Vector Machines — Interview Prep

## Q&A

1. **Q: What is the SVM objective?**
   **A:** Find a separating hyperplane with maximum margin, with soft-margin penalties for violations when data is not separable.

2. **Q: What is a support vector?**
   **A:** A training example on or inside the margin that influences the learned boundary.

3. **Q: What is hinge loss?**
   **A:** `max(0, 1 - y f(x))`, a loss that penalizes examples that are misclassified or inside the margin.

4. **Q: What does `C` control?**
   **A:** The tradeoff between margin width and penalty for margin violations.

5. **Q: Why maximize margin?**
   **A:** A larger margin usually improves robustness and generalization by keeping the boundary away from training examples.

6. **Q: What is the kernel trick?**
   **A:** Computing inner products in an implicit feature space using a kernel function.

7. **Q: Why can kernel SVMs be expensive?**
   **A:** They depend on pairwise kernel computations and can require many support vectors at prediction time.

8. **Q: Do SVMs output probabilities?**
   **A:** Not inherently. They output decision scores; probabilities require calibration such as Platt scaling.

9. **Q: When is a linear SVM a good choice?**
   **A:** High-dimensional sparse classification, especially when ranking or classification margin matters more than calibrated probability.

## Explain it like a principal

SVMs are margin machines. Their value is the discipline of ignoring easy examples and focusing model capacity on boundary-defining cases. In modern systems, they are most relevant as efficient linear classifiers for sparse features and as conceptual grounding for margin losses. A strong answer distinguishes hinge loss from log loss, explains support vectors, and knows why kernel methods are elegant but often operationally expensive.

## Gotchas & follow-ups

- **"All misclassified points affect the model equally."** Hinge loss grows linearly for violations; correctly classified points beyond the margin contribute zero.
- **"SVM scores are probabilities."** They are distances or decision scores, not calibrated probabilities.
- **"Kernels make SVMs scalable."** Kernels add flexibility but usually hurt scaling.
- **Follow-up:** How would you choose between linear SVM, logistic regression, and boosted trees for sparse text classification?
