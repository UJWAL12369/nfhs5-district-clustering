# NFHS District Health Clusters

Unsupervised clustering of Indian districts using indicators from the **National Family Health Survey (NFHS-5)**. The pipeline cleans the raw district-level survey data, reduces redundant features, and groups districts into six health/development clusters using K-Means, with PCA used for visualization.

## What this does

`nfhs_analysis.py` takes the raw NFHS-5 district file (`NFHS_5.csv`) and:

1. **Cleans the data**
   - Strips whitespace from column names
   - Converts NFHS's suppressed-value markers (`*`) to `NaN`
   - Coerces indicator columns to numeric
   - Drops columns with more than 30% missing values
   - Imputes remaining missing values with the column median

2. **Reduces features**
   - Computes the pairwise correlation matrix over all numeric indicators
   - Drops one column from any pair with correlation > 0.90, to avoid redundant/co-linear inputs to the clustering step

3. **Clusters districts**
   - Standardizes features with `StandardScaler`
   - Runs `KMeans` with `k = 6` (chosen via the elbow method — see commented-out code) and `random_state = 42` for reproducibility
   - Assigns each district a `Cluster` (0–5) and a human-readable `Cluster_Name`

4. **Visualizes results**
   - Projects the standardized features to 2 components with PCA (`PCA1`, `PCA2`) for plotting
   - Produces a PCA scatter plot colored by cluster
   - Produces a bar chart of district counts per cluster

5. **Exports results**
   - Writes the final district-level table, including `Cluster`, `Cluster_Name`, `PCA1`, and `PCA2`, to `NFHS_District_Clusters.csv`

## Cluster labels

| Cluster | Name |
|---|---|
| 0 | Moderate Health Districts |
| 1 | Nutritionally Vulnerable Districts |
| 2 | Lifestyle & Health Risk Districts |
| 3 | Urban Transition Districts |
| 4 | High Child Health Risk Districts |
| 5 | High Performing Health Districts |

These labels are qualitative interpretations of each cluster's average indicator profile (`cluster_summary` in the script), not an official NFHS/DGFT classification.

## Getting started

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Usage

1. Place the raw NFHS-5 district-level CSV in the repo root as `NFHS_5.csv`.
2. Run the script:

```bash
python nfhs_analysis.py
```

3. Output: `NFHS_District_Clusters.csv`, plus the PCA scatter plot and cluster-count bar chart shown on screen.

## Repository structure
.
├── nfhs_analysis.py                    # cleaning, feature selection, clustering, PCA
├── NFHS_5.csv                          # raw input data (not included — add your own)
├── NFHS_District_Clusters.csv          # output: districts with cluster assignments
└── india_district_clusters_map.html    # interactive district-level map of the clusters

## Notes / caveats

- Correlation-based feature dropping and the 30%-missingness column cutoff are threshold choices made in this script; adjust them if you want a stricter or looser feature set.
- `k = 6` was chosen via the elbow method (see the commented-out code in the script) rather than a fixed assumption — re-run that block if you'd like to re-validate the choice on updated data.
- Cluster numbering (0–5) is arbitrary and can change between runs/seeds; always join on `Cluster_Name` or re-derive it if you change `random_state` or `k`.


