# Vectors and Norms — Mini-Project: Embedding Metric Bake-Off

Build a small synthetic retrieval experiment that compares dot product, cosine similarity, and Euclidean distance.

## Goal

Create clustered synthetic embeddings, rank documents for a few query vectors, and analyze how metric choice changes retrieval results. The project should make vector magnitude effects visible.

## Dataset

Generate synthetic vectors inside the script:

```python
import torch

torch.manual_seed(0)
n_docs, dim = 200, 16
centers = torch.randn(4, dim)
labels = torch.arange(n_docs) % 4
docs = centers[labels] + 0.15 * torch.randn(n_docs, dim)

# Add magnitude variation that can distort dot-product retrieval.
length_scale = torch.linspace(0.5, 3.0, n_docs).unsqueeze(1)
docs = docs * length_scale

queries = centers + 0.05 * torch.randn(4, dim)
```

No external data is required.

## Implementation Tasks

Create a future file such as `mini_project/metric_bakeoff.py` and implement:

1. L2 norm, pairwise dot product, cosine similarity, and Euclidean distance.
2. Top-5 retrieval for each query under each metric.
3. A simple accuracy measure: how many top-5 documents share the query's cluster label.
4. A report of average document norm among retrieved results for each metric.
5. A near-zero vector case that demonstrates why cosine needs epsilon.

## Expected Workflow

After creating the script, run it from this topic directory:

```bash
uv run python mini_project/metric_bakeoff.py
```

## Expected Outputs

The script should print:

- Top-5 document indices and labels for each query and metric.
- Top-5 cluster-match accuracy by metric.
- Average norm of retrieved documents by metric.
- A short warning or handled result for the near-zero cosine case.

Dot product should tend to retrieve larger-norm documents. Cosine should focus more strongly on cluster direction.

## Writeup Prompt

Write 5-8 sentences answering:

1. Which metric performed best on cluster-match accuracy?
2. Did dot product over-rank large-norm documents?
3. Would you choose the same metric for production semantic search?
4. What offline evaluation would you require before changing a retrieval metric?

## Optional Extensions

- Add normalized Euclidean distance and compare it to cosine similarity.
- Vary the magnitude scale and plot metric sensitivity.
- Add random distractor documents with very large norms.
