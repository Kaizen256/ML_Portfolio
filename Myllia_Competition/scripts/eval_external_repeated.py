from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from official_metric import official_score_parts

def proxy_weights_from_deltas(D: np.ndarray) -> np.ndarray:
    w = np.mean(np.abs(D), axis=0).astype(np.float64)
    w = w / (np.mean(w) + 1e-12)  # mean=1
    return w

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_training_deltas(means_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(means_path)
    gene_cols = [c for c in df.columns if c != "pert_symbol"]

    base_mask = df["pert_symbol"].astype(str) == "non-targeting"
    if base_mask.sum() != 1:
        raise ValueError(f"Expected exactly 1 non-targeting row, found {int(base_mask.sum())}")

    x_base = df.loc[base_mask, gene_cols].iloc[0].to_numpy(np.float32)
    df_train = df.loc[~base_mask].reset_index(drop=True)
    train_genes = df_train["pert_symbol"].astype(str).tolist()

    D = df_train[gene_cols].to_numpy(np.float32) - x_base[None, :]
    return train_genes, gene_cols, D

def load_union_genes(ext_dir: Path, fallback_from_data: list[str] | None = None) -> list[str]:
    p = ext_dir / "union_genes.npy"
    if p.exists():
        return np.load(p, allow_pickle=True).astype(str).tolist()

    if fallback_from_data is None:
        raise FileNotFoundError(
            f"{p} not found. This is not optional. Without it, your external rows can be misaligned."
        )

    print(f"[warn] {p} not found. Rebuilding union via sorted(set(...)).")
    print("[warn] If your external .npy files were built with a different union ordering, rows WILL be misaligned.")
    union = sorted(set(fallback_from_data))
    np.save(p, np.array(union, dtype=object))
    return union

def build_X_for_train_genes(
    train_genes: list[str],
    union_genes: list[str],
    mats: list[np.ndarray],
) -> np.ndarray:
    idx = {g.upper(): i for i, g in enumerate(union_genes)}
    # fallback: mean row per source
    fallbacks = [M.mean(axis=0) for M in mats]

    X_rows = []
    for g in train_genes:
        gi = idx.get(g.upper(), None)
        parts = []
        for M, fb in zip(mats, fallbacks):
            parts.append(M[gi] if gi is not None else fb)
        X_rows.append(np.concatenate(parts, axis=0))
    return np.asarray(X_rows, np.float32)

def repeated_holdout_eval(
    X: np.ndarray,
    D: np.ndarray,
    alpha: float,
    repeats: int,
    test_frac: float,
    seed: int,
):
    n = D.shape[0]
    h = max(1, int(round(test_frac * n)))
    rng = np.random.default_rng(seed)

    w_gene = proxy_weights_from_deltas(D).astype(np.float64)

    scores, sums, wcoses = [], [], []

    for _ in range(repeats):
        perm = rng.permutation(n)
        te = perm[:h]
        tr = perm[h:]

        Xtr, Xte = X[tr], X[te]
        Dtr, Dte = D[tr], D[te]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        model = Ridge(alpha=alpha, fit_intercept=True, random_state=seed)
        model.fit(Xtr_s, Dtr)
        pred = model.predict(Xte_s).astype(np.float32)

        base = Dtr.mean(axis=0).astype(np.float32)
        w = np.tile(w_gene[None, :], (len(te), 1))

        baseline_wmae = np.mean(np.abs(Dte - base[None, :]) * w, axis=1)

        total, sum_terms, wcos, _, _ = official_score_parts(
            y_true=Dte,
            y_pred=pred,
            w=w,
            baseline_wmae=baseline_wmae,
        )
        scores.append(total)
        sums.append(sum_terms)
        wcoses.append(wcos)

    scores = np.array(scores, np.float64)
    sums = np.array(sums, np.float64)
    wcoses = np.array(wcoses, np.float64)

    return {
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "sum_terms_mean": float(sums.mean()),
        "sum_terms_std": float(sums.std(ddof=1)) if len(sums) > 1 else 0.0,
        "wcos_mean": float(wcoses.mean()),
        "wcos_std": float(wcoses.std(ddof=1)) if len(wcoses) > 1 else 0.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()

    ROOT = repo_root()
    means_path = ROOT / "data" / "training_data_means.csv"
    ext_dir = ROOT / "features" / "external"

    train_genes, gene_cols, D = load_training_deltas(means_path)
    union_genes = load_union_genes(ext_dir, fallback_from_data=(gene_cols + train_genes))

    genept_m3 = np.load(ext_dir / "genept_m3_pca128.npy").astype(np.float32)
    genept_ada = np.load(ext_dir / "genept_ada_pca128.npy").astype(np.float32)
    string_g  = np.load(ext_dir / "string_graph_feats.npy").astype(np.float32)
    go_svd    = np.load(ext_dir / "go_svd128.npy").astype(np.float32)
    react_svd = np.load(ext_dir / "reactome_svd128.npy").astype(np.float32)

    combos = {
        "m3_only":        [genept_m3],
        "m3+string":      [genept_m3, string_g],
        "m3+go":          [genept_m3, go_svd],
        "m3+reactome":    [genept_m3, react_svd],
        "m3+go+reactome": [genept_m3, go_svd, react_svd],
        "m3+all":         [genept_m3, genept_ada, string_g, go_svd, react_svd],
    }

    print(f"[info] n_train_perts={len(train_genes)}  n_outputs={D.shape[1]}  alpha={args.alpha}")
    print(f"[info] repeats={args.repeats}  test_frac={args.test_frac}")

    for name, mats in combos.items():
        X = build_X_for_train_genes(train_genes, union_genes, mats)
        out = repeated_holdout_eval(
            X=X, D=D,
            alpha=args.alpha,
            repeats=args.repeats,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        print(
            f"{name:14s}  "
            f"score={out['score_mean']:.5f}±{out['score_std']:.5f}  "
            f"sum_terms={out['sum_terms_mean']:.3f}±{out['sum_terms_std']:.3f}  "
            f"wcos={out['wcos_mean']:.3f}±{out['wcos_std']:.3f}"
        )

if __name__ == "__main__":
    main()
