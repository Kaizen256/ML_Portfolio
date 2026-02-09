from __future__ import annotations

from dataclasses import dataclass
import numpy as np

def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smoothstep easing: t^2 * (3 - 2t) for t in [0, 1]."""
    return t * t * (3.0 - 2.0 * t)

def _gate_smoothstep(x: np.ndarray, a: float = 0.0, b: float = 0.2) -> np.ndarray:
    """
    Gate weights in [0, 1] using a smoothstep ramp from [a, b].

    x <= a -> 0
    x >= b -> 1
    else   -> smoothstep((x-a)/(b-a))
    """
    x = np.asarray(x, dtype=np.float32)
    if b <= a:
        raise ValueError("gate_smoothstep requires b > a")
    t = (x - a) / (b - a)
    t = np.clip(t, 0.0, 1.0)
    return _smoothstep(t).astype(np.float32)


@dataclass(frozen=True)
class MylliaMetricResult:
    score: float
    wcos: float
    pred_wmae: float
    base_wmae: float
    wmae_ratio: float
    sum_terms: float
    mean_term: float


def _weighted_cosine(a: np.ndarray, b: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Weighted cosine per row:
      cos_i = <w*a_i, w*b_i> / (||w*a_i|| * ||w*b_i||)
    Shapes:
      a, b: (N, G)
      w:    (N, G)
    Returns:
      (N,)
    """
    wa = w * a
    wb = w * b
    num = np.sum(wa * wb, axis=1)
    da = np.sqrt(np.sum(wa * wa, axis=1))
    db = np.sqrt(np.sum(wb * wb, axis=1))
    denom = np.maximum(da * db, eps)
    return (num / denom).astype(np.float32)


def _weighted_mae(err: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Weighted MAE per row:
      mae_i = sum_j w_ij * |err_ij| / sum_j w_ij
    """
    num = np.sum(w * np.abs(err), axis=1)
    den = np.maximum(np.sum(w, axis=1), eps)
    return (num / den).astype(np.float32)


def myllia_score(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    *,
    gene_gate_a: float = 0.0,
    gene_gate_b: float = 0.2,
    ratio_gate_a: float = 0.0,
    ratio_gate_b: float = 0.2,
    log_clip: float = 4.0,
    eps: float = 1e-12,
) -> MylliaMetricResult:
    """
    Practical local clone of the competition metric structure:
      - gene-wise weighting via smoothstep gate on |delta_true|
      - weighted cosine similarity (direction)
      - weighted MAE ratio against a zero-delta baseline (scale)
      - log2 term on ratio (better-than-baseline => positive)
      - smooth gate on improvement (1 - ratio), saturating at 20% improvement
      - final score = wcos_mean * mean( gate * log_term )
    """
    dt = np.asarray(delta_true, dtype=np.float32)
    dp = np.asarray(delta_pred, dtype=np.float32)

    if dt.shape != dp.shape:
        raise ValueError(f"Shape mismatch: true={dt.shape} pred={dp.shape}")
    if dt.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N, G); got ndim={dt.ndim}")

    # Gene weights: emphasize genes with meaningful signal magnitude
    w_gene = _gate_smoothstep(np.abs(dt), a=gene_gate_a, b=gene_gate_b)

    # Weighted cosine (per sample then mean)
    wcos_per = _weighted_cosine(dt, dp, w_gene, eps=eps)
    wcos = float(np.mean(wcos_per))

    # WMAE and ratio vs zero-delta baseline
    pred_wmae_per = _weighted_mae(dp - dt, w_gene, eps=eps)
    base_wmae_per = _weighted_mae(dt, w_gene, eps=eps)  # baseline predicts 0, so err = dt

    pred_wmae = float(np.mean(pred_wmae_per))
    base_wmae = float(np.mean(base_wmae_per))

    ratio_per = pred_wmae_per / np.maximum(base_wmae_per, eps)
    wmae_ratio = float(np.mean(ratio_per))

    # Improvement gate: only rewards beating baseline, saturates after 20% improvement
    # improvement = 1 - ratio (so ratio<1 => improvement>0)
    improvement = 1.0 - ratio_per
    gate = _gate_smoothstep(improvement, a=ratio_gate_a, b=ratio_gate_b)

    # Log term: positive if ratio<1, negative if ratio>1
    log_term = -np.log2(np.clip(ratio_per, eps, None))
    if log_clip is not None:
        log_term = np.clip(log_term, -float(log_clip), float(log_clip))

    term = gate * log_term
    mean_term = float(np.mean(term))
    sum_terms = float(np.sum(term))

    score = float(wcos * mean_term)

    return MylliaMetricResult(
        score=score,
        wcos=wcos,
        pred_wmae=pred_wmae,
        base_wmae=base_wmae,
        wmae_ratio=wmae_ratio,
        sum_terms=sum_terms,
        mean_term=mean_term,
    )
