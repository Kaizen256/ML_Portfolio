import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features" / "external"
MEANS = ROOT / "data" / "training_data_means.csv"
VALMAP = ROOT / "data" / "pert_ids_val.csv"

UNION_PATH = FEAT / "union_genes.npy"

SEED = 6
K_OUT = 32


def load_problem():
    df = pd.read_csv(MEANS)

    gene_cols = [c for c in df.columns if c != "pert_symbol"]

    base_row = df.loc[df["pert_symbol"].astype(str) == "non-targeting", gene_cols]
    if len(base_row) == 0:
        raise ValueError("Could not find pert_symbol == 'non-targeting' row in training_data_means.csv")
    base = base_row.iloc[0].to_numpy(dtype=np.float64)

    tr = df.loc[df["pert_symbol"].astype(str) != "non-targeting"].reset_index(drop=True)
    train_genes = tr["pert_symbol"].astype(str).tolist()

    D = tr[gene_cols].to_numpy(dtype=np.float64) - base[None, :]
    return train_genes, D, gene_cols


def weighted_cosine(D_true, D_pred, w):
    A = D_true * w[None, :]
    B = D_pred * w[None, :]
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float(np.mean(num / den))


def load_union(gene_cols, train_genes):
    if UNION_PATH.exists():
        union = np.load(UNION_PATH, allow_pickle=True)
        union = [str(x) for x in union.tolist()]
        return union

    # Fallback (not ideal): rebuild union deterministically from what we can see
    val_perts = pd.read_csv(VALMAP)["pert"].astype(str).tolist()
    union = sorted(set(gene_cols) | set(train_genes) | set(val_perts))
    return union

def load_saved_union_or_die(feat_dir):
    p = feat_dir / "union_genes.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run engineer_external_features.py first "
            "or re-run it after cleaning external/features."
        )
    return np.load(p, allow_pickle=True).tolist()

def make_ridx(train_genes):
    union = load_saved_union_or_die(FEAT)
    u2i = {g.upper(): i for i, g in enumerate(union)}
    missing = [g for g in train_genes if g.upper() not in u2i]
    if missing:
        raise ValueError(f"Missing {len(missing)} train genes from union (examples: {missing[:10]})")
    ridx = np.array([u2i[g.upper()] for g in train_genes], dtype=np.int64)
    return ridx


def load_blocks(ridx):
    blocks = {
        "genept_m3": np.load(FEAT / "genept_m3_pca128.npy")[ridx],
        "genept_ada": np.load(FEAT / "genept_ada_pca128.npy")[ridx],
        "string_graph": np.load(FEAT / "string_graph_feats.npy")[ridx],
        "go_svd": np.load(FEAT / "go_svd128.npy")[ridx],
        "reactome_svd": np.load(FEAT / "reactome_svd128.npy")[ridx],
    }
    # ensure float64 for stability
    for k in list(blocks.keys()):
        blocks[k] = np.asarray(blocks[k], dtype=np.float64)
    return blocks


def build_groups(X_for_groups, n_samples):
    n_clusters = min(8, max(2, n_samples // 8))
    groups = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto").fit_predict(X_for_groups)

    uniq = np.unique(groups)
    if len(uniq) < 2:
        groups = np.arange(n_samples, dtype=np.int64)

    return groups


def cv_score_combo_alpha(X, D, groups, alpha):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    scores = []
    for tr_idx, va_idx in gkf.split(X, groups=groups):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_va = sc.transform(X[va_idx])

        svd = TruncatedSVD(n_components=K_OUT, random_state=SEED)
        Y_tr = svd.fit_transform(D[tr_idx])

        w = (np.mean(np.abs(D[tr_idx]), axis=0) + 1e-6)
        w = w / w.mean()

        m = Ridge(alpha=alpha, solver="svd", fit_intercept=True)
        m.fit(X_tr, Y_tr)

        Yhat = m.predict(X_va)
        Dhat = svd.inverse_transform(Yhat)

        scores.append(weighted_cosine(D[va_idx], Dhat, w))

    return float(np.mean(scores)), float(np.std(scores))


def main():
    train_genes, D, gene_cols = load_problem()

    union = load_union(gene_cols, train_genes)
    ridx = make_ridx(train_genes)
    blocks = load_blocks(ridx)

    n = len(train_genes)
    groups = build_groups(blocks["genept_m3"], n_samples=n)

    combos = {
        "m3_only": ["genept_m3"],
        "m3+string": ["genept_m3", "string_graph"],
        "m3+go": ["genept_m3", "go_svd"],
        "m3+reactome": ["genept_m3", "reactome_svd"],
        "m3+go+reactome": ["genept_m3", "go_svd", "reactome_svd"],
        "m3+all": ["genept_m3", "genept_ada", "string_graph", "go_svd", "reactome_svd"],
    }

    alphas = np.logspace(-4, 7, 12)

    for name, keys in combos.items():
        X = np.concatenate([blocks[k] for k in keys], axis=1)

        best_mu = -1e18
        best_sd = None
        best_a = None

        for a in alphas:
            mu, sd = cv_score_combo_alpha(X, D, groups, alpha=float(a))
            if mu > best_mu:
                best_mu, best_sd, best_a = mu, sd, float(a)

        print(f"{name:14s}  best_mean={best_mu:.5f}  std={best_sd:.5f}  alpha={best_a:g}  dims={X.shape[1]}")


if __name__ == "__main__":
    main()
