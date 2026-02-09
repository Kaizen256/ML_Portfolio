import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features" / "external"

def block_stats(name, X):
    # variance per feature, and how many columns are ~constant
    v = X.var(axis=0)
    const_frac = float((v < 1e-12).mean())
    nan_frac = float(np.isnan(X).mean())
    return {
        "shape": list(X.shape),
        "nan_frac": nan_frac,
        "const_feature_frac": const_frac,
        "mean_abs": float(np.mean(np.abs(X))),
        "mean_var": float(np.mean(v)),
    }

def main():
    blocks = {
        "genept_m3": np.load(FEAT / "genept_m3_pca128.npy"),
        "genept_ada": np.load(FEAT / "genept_ada_pca128.npy"),
        "string_graph": np.load(FEAT / "string_graph_feats.npy"),
        "go_svd": np.load(FEAT / "go_svd128.npy"),
        "reactome_svd": np.load(FEAT / "reactome_svd128.npy"),
    }

    out = {k: block_stats(k, v) for k, v in blocks.items()}

    cov = json.loads((FEAT / "coverage_report.json").read_text())
    out["coverage_report"] = cov

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
