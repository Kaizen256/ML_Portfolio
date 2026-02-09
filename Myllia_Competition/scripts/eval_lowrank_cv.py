import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEAT = ROOT / "features" / "external"


# -----------------------------
# Metric (official formula pieces)
# -----------------------------
class ParticipantVisibleError(Exception):
    pass


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _gate_smoothstep(x: np.ndarray, a: float = 0.0, b: float = 0.2) -> np.ndarray:
    if b <= a:
        raise ValueError("gate_smoothstep requires b > a")
    t = (x - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return _smoothstep(t)


def weighted_cosine_smooth(
    a: np.ndarray,
    b: np.ndarray,
    left: float = 0.0,
    right: float = 0.2,
    eps: float = 1e-12,
) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("weighted_cosine: a and b must have the same length")

    x = np.maximum(np.abs(a), np.abs(b))
    w = _gate_smoothstep(x, left, right)
    w2 = w * w

    num = np.sum(w2 * a * b)
    den_a = np.sqrt(np.sum(w2 * a * a))
    den_b = np.sqrt(np.sum(w2 * b * b))
    den = den_a * den_b

    if den < eps:
        return 0.0
    return float(num / den)


def score_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w: np.ndarray,
    baseline_wmae: np.ndarray,
    eps: float = 1e-12,
    max_log2: float = 5.0,
    cos_left: float = 0.0,
    cos_right: float = 0.2,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    baseline_wmae = np.asarray(baseline_wmae, dtype=np.float64)

    abs_err = np.abs(y_true - y_pred)
    pred_wmae = np.mean(abs_err * w, axis=1)

    pred_wmae_safe = np.maximum(pred_wmae, eps)
    baseline_safe = np.maximum(baseline_wmae, eps)

    terms = np.log2(baseline_safe / pred_wmae_safe)
    terms = np.minimum(terms, max_log2)

    sum_wmae = float(np.sum(terms))
    wcos = weighted_cosine_smooth(
        y_pred.ravel(), y_true.ravel(), left=cos_left, right=cos_right, eps=eps
    )
    final_score = float(sum_wmae * max(0.0, wcos))

    return {
        "final_score": float(round(final_score, 5)),
        "sum_wmae": float(sum_wmae),
        "wcos": float(wcos),
        "pred_wmae_mean": float(pred_wmae.mean()),
        "pred_wmae_median": float(np.median(pred_wmae)),
        "baseline_wmae_mean": float(baseline_wmae.mean()),
        "baseline_wmae_median": float(np.median(baseline_wmae)),
        "term_mean": float(terms.mean()),
        "term_median": float(np.median(terms)),
        "term_pos_frac": float((terms > 0).mean()),
    }


# -----------------------------
# Import model utilities from your training script
# -----------------------------
from model4_script.train_lowrank import (
    load_means,
    _load_feature_matrix,
    _cosine_kernel_predict,
)


# -----------------------------
# Proxy arrays (no pandas, no fragmentation)
# -----------------------------
def build_proxy_arrays_from_training_means(
    base: np.ndarray,
    D: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Proxy mode when you do not have the host-format solution file.

    y_true = base + D
    w = ones (sums to n_genes per row, matches the host sanity rule)
    baseline_wmae = mean(|y_true - base|) per row (proxy baseline)
    """
    y_true = (D + base[None, :]).astype(np.float64)
    n_rows, n_genes = y_true.shape
    w = np.ones((n_rows, n_genes), dtype=np.float64)
    baseline_wmae = np.mean(np.abs(y_true - base[None, :]), axis=1).astype(np.float64)
    return y_true, w, baseline_wmae


def infer_gene_cols_from_solution(sol: pd.DataFrame, baseline_col: str = "baseline_wmae", weight_prefix: str = "w_") -> List[str]:
    genes = []
    for c in sol.columns:
        if c == "pert_id" or c == baseline_col:
            continue
        if c.startswith(weight_prefix):
            continue
        if f"{weight_prefix}{c}" in sol.columns:
            genes.append(c)
    if not genes:
        raise ValueError("Could not infer gene columns from solution (need gene + matching w_<gene>).")
    return genes


def load_exact_solution_arrays(
    solution_path: Path,
    pert_ids: List[str],
    gene_cols_train: List[str],
    base_train: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the host-format solution file and returns:
      gene_cols, base_aligned, y_true_all, w_all, baseline_all
    aligned to pert_ids row order and gene_cols order.
    """
    sol = pd.read_csv(solution_path)
    if "pert_id" not in sol.columns:
        raise ValueError("Solution file must contain pert_id column.")
    if "baseline_wmae" not in sol.columns:
        raise ValueError("Solution file must contain baseline_wmae column.")

    gene_cols = infer_gene_cols_from_solution(sol)
    weight_cols = [f"w_{g}" for g in gene_cols]

    missing_w = [c for c in weight_cols if c not in sol.columns]
    if missing_w:
        raise ValueError(f"Solution file missing weight columns (example: {missing_w[0]})")

    # Ensure these genes exist in training_means gene set so base can be aligned
    train_set = set(gene_cols_train)
    missing_genes = [g for g in gene_cols if g not in train_set]
    if missing_genes:
        raise ValueError(f"Solution has genes not in training_data_means.csv (example: {missing_genes[0]})")

    # Align base to solution gene order
    idx = [gene_cols_train.index(g) for g in gene_cols]
    base = base_train[idx].astype(np.float64)

    # Align rows
    sol["pert_id"] = sol["pert_id"].astype(str)
    sol = sol.set_index("pert_id")

    missing_ids = [pid for pid in pert_ids if pid not in sol.index]
    if missing_ids:
        raise ValueError(f"Solution file missing {len(missing_ids)} ids (example: {missing_ids[0]})")

    sol = sol.loc[pert_ids]

    y_true_all = sol[gene_cols].to_numpy(np.float64)
    w_all = sol[weight_cols].to_numpy(np.float64)
    baseline_all = sol["baseline_wmae"].to_numpy(np.float64)

    # Sanity: weights must sum to n_genes per row
    n_genes = w_all.shape[1]
    row_sums = np.sum(w_all, axis=1)
    if not np.allclose(row_sums, n_genes, atol=1e-6, rtol=0.0):
        raise ValueError("Solution weights must sum to n_genes per row (host rule).")

    if (w_all < 0).any():
        raise ValueError("Solution weights must be non-negative.")

    if not np.isfinite(y_true_all).all() or not np.isfinite(w_all).all() or not np.isfinite(baseline_all).all():
        raise ValueError("Solution contains NaN/inf values.")

    return gene_cols, base, y_true_all, w_all, baseline_all


# -----------------------------
# CV runner
# -----------------------------
def run_cv(
    k: int,
    alpha: float,
    tau: float,
    topk: int,
    blend: float,
    norm_match: bool,
    repeats: int,
    test_frac: float,
    seed: int,
    solution_path: Optional[Path],
    eps: float,
    max_log2: float,
    cos_left: float,
    cos_right: float,
) -> None:
    np.random.seed(seed)

    base_train, D_train, train_genes, gene_cols_train = load_means(DATA / "training_data_means.csv")
    pert_ids = [str(g) for g in train_genes]

    # Determine scoring arrays and gene ordering
    if solution_path is not None and solution_path.exists():
        gene_cols, base, y_true_all, w_all, baseline_all = load_exact_solution_arrays(
            solution_path=solution_path,
            pert_ids=pert_ids,
            gene_cols_train=gene_cols_train,
            base_train=base_train,
        )
        print(f"[info] Using exact solution file: {solution_path}")
    else:
        gene_cols = gene_cols_train
        base = base_train.astype(np.float64)
        D_train = D_train.astype(np.float64)
        y_true_all, w_all, baseline_all = build_proxy_arrays_from_training_means(base, D_train)
        print("[warn] No solution file found. Using proxy weights=1 and proxy baseline_wmae from base.")

    # Features for the 80 training genes
    genes_u = [g.strip().upper() for g in train_genes]
    X_raw, _, _ = _load_feature_matrix(genes_u, FEAT, verbose=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    n = y_true_all.shape[0]
    n_test = max(1, int(test_frac * n))

    fold_stats: List[Dict[str, float]] = []

    for r in range(repeats):
        idx = np.random.permutation(n)
        te = idx[:n_test]
        tr = idx[n_test:]

        # Train in delta space on training fold
        D_tr = (y_true_all[tr] - base[None, :]).astype(np.float32)

        X_tr = X[tr]
        X_te = X[te]
        X_tr_raw = X_raw[tr].astype(np.float32)
        X_te_raw = X_raw[te].astype(np.float32)

        kk = max(1, min(k, min(D_tr.shape[0] - 1, D_tr.shape[1])))

        svd = TruncatedSVD(n_components=kk, random_state=seed + r)
        C_tr = svd.fit_transform(D_tr).astype(np.float32)
        V = svd.components_.astype(np.float32)

        ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=seed + r)
        ridge.fit(X_tr, C_tr)
        C_r = ridge.predict(X_te).astype(np.float32)

        C_k = _cosine_kernel_predict(
            Q=X_te_raw,
            K=X_tr_raw,
            V=C_tr,
            tau=tau,
            topk=topk,
        ).astype(np.float32)

        C_pred = (blend * C_r + (1.0 - blend) * C_k).astype(np.float32)
        D_pred = (C_pred @ V).astype(np.float32)

        if norm_match:
            target_norm = float(np.median(np.linalg.norm(D_tr, axis=1)))
            pred_norm = np.linalg.norm(D_pred, axis=1) + 1e-12
            D_pred = D_pred * (target_norm / pred_norm)[:, None]

        # Score in expression space
        y_pred = (D_pred + base[None, :]).astype(np.float64)
        y_true = y_true_all[te].astype(np.float64)
        w = w_all[te].astype(np.float64)
        baseline = baseline_all[te].astype(np.float64)

        stats = score_components(
            y_true=y_true,
            y_pred=y_pred,
            w=w,
            baseline_wmae=baseline,
            eps=eps,
            max_log2=max_log2,
            cos_left=cos_left,
            cos_right=cos_right,
        )
        fold_stats.append(stats)

    # Summaries
    def agg(key: str) -> Tuple[float, float]:
        arr = np.array([d[key] for d in fold_stats], dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    keys = [
        "final_score",
        "sum_wmae",
        "wcos",
        "pred_wmae_mean",
        "pred_wmae_median",
        "baseline_wmae_mean",
        "baseline_wmae_median",
        "term_mean",
        "term_median",
        "term_pos_frac",
    ]

    print("\n[cv] Summary over repeats")
    for kname in keys:
        m, s = agg(kname)
        if kname == "term_pos_frac":
            print(f"  {kname:20s}: {m:.3f} ± {s:.3f}")
        else:
            print(f"  {kname:20s}: {m:.5f} ± {s:.5f}")

    m_score, _ = agg("final_score")
    m_sum, _ = agg("sum_wmae")
    m_wcos, _ = agg("wcos")

    print("\n[cv] Diagnosis")
    if m_wcos <= 0.01:
        print("  wcos is near zero. The cosine gate is nuking your score.")
    if m_sum <= 0.0:
        print("  sum_wmae is non-positive. You are not beating baseline_wmae on average (terms <= 0).")
    if m_sum > 0.0 and m_wcos > 0.0:
        print("  Both components are positive. Now tune for bigger sum_wmae without crashing wcos.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--norm-match", action="store_true")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=6)

    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--max-log2", type=float, default=5.0)
    ap.add_argument("--cos-left", type=float, default=0.0)
    ap.add_argument("--cos-right", type=float, default=0.2)

    ap.add_argument("--solution", type=str, default="", help="Optional host-format solution CSV with weights + baseline_wmae")

    args = ap.parse_args()
    sol_path = Path(args.solution) if args.solution else None

    run_cv(
        k=args.k,
        alpha=args.alpha,
        tau=args.tau,
        topk=args.topk,
        blend=args.blend,
        norm_match=args.norm_match,
        repeats=args.repeats,
        test_frac=args.test_frac,
        seed=args.seed,
        solution_path=sol_path,
        eps=args.eps,
        max_log2=args.max_log2,
        cos_left=args.cos_left,
        cos_right=args.cos_right,
    )


if __name__ == "__main__":
    main()
