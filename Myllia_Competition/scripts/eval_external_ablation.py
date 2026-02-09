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

SEED = 6
K_OUT = 32

def load_problem():
    df = pd.read_csv(MEANS)
    gene_cols = [c for c in df.columns if c != "pert_symbol"]
    base = df.loc[df["pert_symbol"].astype(str) == "non-targeting", gene_cols].iloc[0].to_numpy(np.float32)
    tr = df.loc[df["pert_symbol"].astype(str) != "non-targeting"].reset_index(drop=True)
    train_genes = tr["pert_symbol"].astype(str).tolist()
    D = tr[gene_cols].to_numpy(np.float32) - base[None, :]
    return train_genes, D, gene_cols

def weighted_cosine(D_true, D_pred, w):
    A = D_true * w[None, :]
    B = D_pred * w[None, :]
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8
    return float(np.mean(num / den))

def main():
    train_genes, D, gene_cols = load_problem()

    # union ordering used by engineer script
    union = sorted(set(gene_cols) | set(train_genes) | set(pd.read_csv(VALMAP)["pert"].astype(str).tolist()))
    u2i = {g.upper(): i for i, g in enumerate(union)}

    def rows_for(genes):
        idx = [u2i[g.upper()] for g in genes]
        return np.array(idx, dtype=np.int64)

    ridx = rows_for(train_genes)

    blocks = {
        "genept_m3": np.load(FEAT / "genept_m3_pca128.npy")[ridx],
        "genept_ada": np.load(FEAT / "genept_ada_pca128.npy")[ridx],
        "string_graph": np.load(FEAT / "string_graph_feats.npy")[ridx],
        "go_svd": np.load(FEAT / "go_svd128.npy")[ridx],
        "reactome_svd": np.load(FEAT / "reactome_svd128.npy")[ridx],
    }

    # output compression
    svd_out = TruncatedSVD(n_components=K_OUT, random_state=SEED)
    Y = svd_out.fit_transform(D).astype(np.float32)

    w = (np.mean(np.abs(D), axis=0) + 1e-6).astype(np.float32)
    w = w / w.mean()

    # hard-ish CV: cluster by embedding (use genept_m3 as "gene space")
    groups = KMeans(n_clusters=min(8, max(2, len(train_genes)//8)),
                    random_state=SEED, n_init="auto").fit_predict(blocks["genept_m3"])
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    def score_with(X):
        Xs = StandardScaler().fit_transform(X).astype(np.float32)
        scores = []
        for tr_idx, va_idx in gkf.split(Xs, groups=groups):
            m = Ridge(alpha=1.0, random_state=SEED)
            m.fit(Xs[tr_idx], Y[tr_idx])
            Yhat = m.predict(Xs[va_idx]).astype(np.float32)
            Dhat = svd_out.inverse_transform(Yhat).astype(np.float32)
            scores.append(weighted_cosine(D[va_idx], Dhat, w))
        return float(np.mean(scores)), float(np.std(scores))

    # Ablations
    combos = {
        "m3_only": ["genept_m3"],
        "m3+string": ["genept_m3", "string_graph"],
        "m3+go": ["genept_m3", "go_svd"],
        "m3+reactome": ["genept_m3", "reactome_svd"],
        "m3+go+reactome": ["genept_m3", "go_svd", "reactome_svd"],
        "m3+all": ["genept_m3", "genept_ada", "string_graph", "go_svd", "reactome_svd"],
    }

    for name, keys in combos.items():
        X = np.concatenate([blocks[k] for k in keys], axis=1)
        mu, sd = score_with(X)
        print(f"{name:14s}  mean={mu:.5f}  std={sd:.5f}  dims={X.shape[1]}")

if __name__ == "__main__":
    main()
