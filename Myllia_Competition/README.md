# External Feature Engineering + Audit + Ablation Summary

## `py engineer_external_features.py`

### Union gene set
- **union_genes:** `5143`

### Coverage (how many union genes got a feature vector)

| Source         | Hits | Coverage |
|----------------|-----:|---------:|
| GenePT (m3)    | 4972 |  96.68%  |
| GenePT (ada)   | 4972 |  96.68%  |
| STRING graph   | 4451 |  86.54%  |
| GO             | 4982 |  96.87%  |
| Reactome       | 3314 |  64.44%  |

### Saved feature artifacts
- **All files written under:** `...\Myllia_Competition\features\external\`

| Artifact                | What it is                                       | Output path |
|-------------------------|--------------------------------------------------|------------|
| `genept_m3_pca128.npy`  | GenePT model-3 text embedding, PCA to 128 dims   | `C:\Users\rowes\Documents\GitHub\2026-ML-Projects\Myllia_Competition\features\external\genept_m3_pca128.npy` |
| `genept_ada_pca128.npy` | GenePT ada text embedding, PCA to 128 dims       | `C:\Users\rowes\Documents\GitHub\2026-ML-Projects\Myllia_Competition\features\external\genept_ada_pca128.npy` |
| `string_graph_feats.npy`| STRING-derived graph features                    | `C:\Users\rowes\Documents\GitHub\2026-ML-Projects\Myllia_Competition\features\external\string_graph_feats.npy` |
| `go_svd128.npy`         | GO features reduced with SVD to 128 dims         | `C:\Users\rowes\Documents\GitHub\2026-ML-Projects\Myllia_Competition\features\external\go_svd128.npy` |
| `reactome_svd128.npy`   | Reactome features reduced with SVD to 128 dims   | `C:\Users\rowes\Documents\GitHub\2026-ML-Projects\Myllia_Competition\features\external\reactome_svd128.npy` |

---

## `py audit_external_sources.py`

### Feature matrix health checks (sanity audit)
All blocks have:
- `nan_frac = 0.0` (no NaNs)
- `const_feature_frac = 0.0` (no constant columns)

| Feature block   | Shape (genes, dims) | NaN frac | Const frac | mean_abs | mean_var |
|----------------|---------------------:|---------:|-----------:|---------:|---------:|
| `genept_m3`     | (5143, 128)          | 0.0      | 0.0        | 0.04112  | 0.00318  |
| `genept_ada`    | (5143, 128)          | 0.0      | 0.0        | 0.02257  | 0.00098  |
| `string_graph`  | (5143, 3)            | 0.0      | 0.0        | 33.51363 | 2400.69531 |
| `go_svd`        | (5143, 128)          | 0.0      | 0.0        | 0.16988  | 0.06709  |
| `reactome_svd`  | (5143, 128)          | 0.0      | 0.0        | 0.04120  | 0.01887  |

### Coverage report (deduped)
Your printout repeated the same `coverage_report` block like it got stuck in a loop. Keeping the meaningful values:

- **union_genes:** `5143`
- **genept_m3_hits:** `4972`
- **genept_ada_hits:** `4972`
- **string_hits:** `4451`
- **go_hits:** `4982`
- **reactome_hits:** `3314`
- **paths:** same as `engineer_external_features.py`

---

## `py eval_external_ablation.py`

### Ablation results

| Feature set        | Dims | Mean    | Std     | Δ vs m3_only |
|-------------------|-----:|--------:|--------:|-------------:|
| `m3_only`          | 128  | 0.15310 | 0.05844 | +0.00000 |
| `m3+string`        | 131  | 0.16039 | 0.05569 | +0.00729 |
| `m3+go`            | 256  | 0.14887 | 0.04279 | -0.00423 |
| `m3+reactome`      | 256  | 0.12254 | 0.03659 | -0.03056 |
| `m3+go+reactome`   | 384  | 0.14391 | 0.03524 | -0.00919 |
| `m3+all`           | 515  | 0.17101 | 0.02872 | +0.01791 |

### What “all” likely includes (matches 515 dims)
- GenePT m3 (128) + GenePT ada (128) + STRING (3) + GO (128) + Reactome (128)  
- **Total:** `128 + 128 + 3 + 128 + 128 = 515`

### Useful takeaways (based strictly on these numbers)
- **Best mean:** `m3+all` (0.17101), and also **lowest std** here (0.02872).
- **STRING helps** a bit when added to m3 alone.
- **Reactome alone hurts a lot** in this setup (big drop vs baseline), even though it has decent coverage for the genes it hits.
