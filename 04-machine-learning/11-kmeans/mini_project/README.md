# K-Means — Mini-Project: Customer Segment Simulator

## Goal

Cluster synthetic customer behavior data and evaluate how feature scaling and `K` selection affect segment quality.

## Dataset

Generate 1,500 synthetic customers with features such as monthly spend, visit frequency, discount usage rate, and tenure. Create hidden segment labels only for evaluation; do not use them during clustering.

## Implementation tasks

Create `mini_project/customer_segments.py` in a future implementation pass. It should:

1. Generate synthetic segment data.
2. Run k-means with and without standardization.
3. Sweep `K` from 2 through 8.
4. Track inertia, silhouette score, and agreement with hidden labels.
5. Print centroid profiles in original feature units.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/customer_segments.py
```

## Expected outputs

- Inertia and silhouette tables.
- Centroid summaries that can be interpreted as customer segments.
- A recommendation for `K` with caveats.

## Writeup prompt

Explain how feature scaling changed the discovered segments and why lower inertia alone was not enough to choose `K`.

## Optional extensions

- Add k-means++ initialization.
- Test cluster stability across random seeds.
- Add PCA visualization.
