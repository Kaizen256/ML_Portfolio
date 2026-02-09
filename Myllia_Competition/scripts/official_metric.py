import numpy as np

def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)

def _gate(x: np.ndarray, left: float = 0.0, right: float = 0.2) -> np.ndarray:
    t = (x - left) / (right - left)
    t = np.clip(t, 0.0, 1.0)
    return _smoothstep(t)

def weighted_cosine(pred_flat, true_flat, left: float = 0.0, right: float = 0.2, eps: float = 1e-12) -> float:
    pred = np.asarray(pred_flat, np.float64).ravel()
    true = np.asarray(true_flat, np.float64).ravel()
    x = np.maximum(np.abs(pred), np.abs(true))
    w = _gate(x, left, right)
    w2 = w * w
    num = np.sum(w2 * pred * true)
    den = np.sqrt(np.sum(w2 * pred * pred)) * np.sqrt(np.sum(w2 * true * true))
    return 0.0 if den < eps else float(num / den)

def official_score_parts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w: np.ndarray,
    baseline_wmae: np.ndarray,
    eps: float = 1e-12,
    max_log2: float = 5.0,
    cos_left: float = 0.0,
    cos_right: float = 0.2,
):
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    w = np.asarray(w, np.float64)
    baseline_wmae = np.asarray(baseline_wmae, np.float64)

    pred_wmae = np.mean(np.abs(y_true - y_pred) * w, axis=1)
    pred_wmae = np.maximum(pred_wmae, eps)
    baseline = np.maximum(baseline_wmae, eps)

    terms = np.log2(baseline / pred_wmae)
    terms = np.minimum(terms, max_log2)
    sum_terms = float(np.sum(terms))

    wcos = weighted_cosine(y_pred.ravel(), y_true.ravel(), left=cos_left, right=cos_right, eps=eps)
    total = float(sum_terms * max(0.0, wcos))
    return total, sum_terms, float(wcos), terms, pred_wmae
