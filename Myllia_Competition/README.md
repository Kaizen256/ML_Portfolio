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


loss = weighted_l1_like_rowweighted(
                dt_b, dp_b, bw_b,
                mode="inv_sqrt",
                clamp_min=0.5,
                clamp_max=3.0,
            )

fold 1: best_score=0.162418 best_alpha=0.700 best_epoch=25
fold 2: best_score=0.111889 best_alpha=0.700 best_epoch=25
fold 3: best_score=0.114223 best_alpha=0.680 best_epoch=25
fold 4: best_score=0.121005 best_alpha=0.700 best_epoch=25
fold 5: best_score=0.156279 best_alpha=0.700 best_epoch=25
fold 6: best_score=0.188296 best_alpha=0.700 best_epoch=25
fold 7: best_score=0.178646 best_alpha=0.640 best_epoch=25
fold 8: best_score=0.123527 best_alpha=0.620 best_epoch=25
cv mean: 0.1445352986908065 std: 0.028539096177756983
median best_epoch = 25
OOF global alpha: 0.699999988079071 OOF score: 0.14320898877057076

loss = weighted_l1_like_rowweighted(
                dt_b, dp_b, bw_b,
                mode="inv_sqrt",
                clamp_min=0.7,
                clamp_max=2.0,
)

fold 1: best_score=0.163247 best_alpha=0.700 best_epoch=25
fold 2: best_score=0.112429 best_alpha=0.700 best_epoch=25
fold 3: best_score=0.111956 best_alpha=0.640 best_epoch=25
fold 4: best_score=0.122694 best_alpha=0.700 best_epoch=25
fold 5: best_score=0.157536 best_alpha=0.700 best_epoch=25
fold 6: best_score=0.187481 best_alpha=0.680 best_epoch=25
fold 7: best_score=0.175097 best_alpha=0.620 best_epoch=25
fold 8: best_score=0.123204 best_alpha=0.620 best_epoch=25
cv mean: 0.14420547200856515 std: 0.02810893605355988
median best_epoch = 25
OOF global alpha: 0.699999988079071 OOF score: 0.1427402023207307

loss = weighted_l1_like_rowweighted(
                dt_b, dp_b, bw_b,
                mode="inv_sqrt",
                clamp_min=0.3,
                clamp_max=5.0,
            )

fold 1: best_score=0.159718 best_alpha=0.700 best_epoch=25
fold 2: best_score=0.111553 best_alpha=0.700 best_epoch=25
fold 3: best_score=0.115490 best_alpha=0.700 best_epoch=25
fold 4: best_score=0.118156 best_alpha=0.700 best_epoch=25
fold 5: best_score=0.151292 best_alpha=0.700 best_epoch=25
fold 6: best_score=0.189359 best_alpha=0.700 best_epoch=25
fold 7: best_score=0.179760 best_alpha=0.660 best_epoch=25
fold 8: best_score=0.126236 best_alpha=0.660 best_epoch=25
cv mean: 0.14394549028394532 std: 0.028472675748411735
median best_epoch = 25
OOF global alpha: 0.699999988079071 OOF score: 0.14277189015750835

loss = weighted_l1_like_rowweighted(
                dt_b, dp_b, bw_b,
                mode="inv_log",
                clamp_min=0.5,
                clamp_max=3.0,
)

fold 1: best_score=0.163073 best_alpha=0.700 best_epoch=25
fold 2: best_score=0.112248 best_alpha=0.700 best_epoch=25
fold 3: best_score=0.111654 best_alpha=0.640 best_epoch=25
fold 4: best_score=0.123021 best_alpha=0.700 best_epoch=25
fold 5: best_score=0.157770 best_alpha=0.700 best_epoch=25
fold 6: best_score=0.187842 best_alpha=0.680 best_epoch=25
fold 7: best_score=0.174498 best_alpha=0.620 best_epoch=25
fold 8: best_score=0.123217 best_alpha=0.620 best_epoch=25
cv mean: 0.14416549925900013 std: 0.028133274757033498
median best_epoch = 25
OOF global alpha: 0.699999988079071 OOF score: 0.14268128816927295

loss = weighted_l1_like_rowweighted(
                dt_b, dp_b, bw_b,
                mode="inv",
                clamp_min=0.5,
                clamp_max=3.0,
)

fold 1: best_score=0.163200 best_alpha=0.700 best_epoch=25
fold 2: best_score=0.112736 best_alpha=0.700 best_epoch=25
fold 3: best_score=0.112199 best_alpha=0.640 best_epoch=25
fold 4: best_score=0.122677 best_alpha=0.700 best_epoch=25
fold 5: best_score=0.157525 best_alpha=0.700 best_epoch=25
fold 6: best_score=0.187082 best_alpha=0.680 best_epoch=25
fold 7: best_score=0.175008 best_alpha=0.620 best_epoch=25
fold 8: best_score=0.123332 best_alpha=0.620 best_epoch=25
cv mean: 0.1442199672741995 std: 0.027927069221901964
median best_epoch = 25
OOF global alpha: 0.699999988079071 OOF score: 0.14277206537857978