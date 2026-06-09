# Decision Trees — Interview Prep

## Q&A

1. **Q: How does a decision tree choose a split?**
   **A:** It evaluates candidate feature thresholds and selects the one with the largest impurity or loss reduction.

2. **Q: What is Gini impurity?**
   **A:** `1 - sum_k p_k^2`, the probability of misclassifying a random example if labeled according to node class frequencies.

3. **Q: Entropy versus Gini?**
   **A:** Both measure class impurity. Entropy is information-theoretic; Gini is simpler and often gives similar splits.

4. **Q: What prediction does a regression tree leaf make?**
   **A:** Usually the mean target of training examples in that leaf.

5. **Q: Why do trees overfit?**
   **A:** Deep trees can create tiny leaves that memorize noise or individual examples.

6. **Q: How do you regularize a tree?**
   **A:** Limit depth, increase minimum samples per leaf, require minimum impurity decrease, prune, or use ensembles.

7. **Q: Do trees need feature scaling?**
   **A:** Usually no. Threshold-based splits are invariant to monotonic feature scaling.

8. **Q: How do trees handle nonlinear interactions?**
   **A:** Later splits are conditional on earlier splits, so different feature rules apply in different regions.

9. **Q: What is a weakness of single decision trees?**
   **A:** High variance and unstable structure under small data changes.

## Explain it like a principal

Decision trees are rule learners that trade smoothness for conditional structure. They are excellent for tabular interactions and human-readable segmentation, but a single tree is rarely the highest-performing production model because variance is high. Mature use means constraining depth, validating leaf support, checking leakage-prone splits, and knowing when to move to forests or boosted trees.

## Gotchas & follow-ups

- **"Feature importance from a tree is causal."** It is not; it reflects split utility in the observed data and can be biased.
- **"A pure leaf means a reliable rule."** A pure leaf with two examples is not reliable.
- **"Trees cannot extrapolate."** Correct for standard trees: predictions are piecewise constant, so regression outside observed ranges is limited.
- **Follow-up:** How would you prevent a tree from splitting on a user ID or timestamp leakage feature?
