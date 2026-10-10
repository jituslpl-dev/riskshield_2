import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def metrics(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    calibration = []
    for i in range(bins):
        mask = (p >= i / bins) & ((p < (i + 1) / bins) if i < bins - 1 else (p <= 1))
        if mask.any():
            calibration.append(
                {
                    "count": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()),
                    "event_rate": float(y[mask].mean()),
                }
            )
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(
            sum(x["count"] * abs(x["mean_probability"] - x["event_rate"]) for x in calibration) / len(y)
        ),
        "calibration_bins": calibration,
    }


def bootstrap_intervals(y, p, repeats=300, seed=42):
    """Stratified bootstrap uncertainty; does not capture temporal dependence."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both classes required for bootstrap.")
    ap = []
    brier = []
    for _ in range(repeats):
        idx = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        ap.append(average_precision_score(y[idx], p[idx]))
        brier.append(brier_score_loss(y[idx], p[idx]))
    return {
        "method": "stratified row bootstrap; temporal dependence not modeled",
        "repeats": repeats,
        "pr_auc_95": list(map(float, np.quantile(ap, [0.025, 0.975]))),
        "brier_95": list(map(float, np.quantile(brier, [0.025, 0.975]))),
    }
