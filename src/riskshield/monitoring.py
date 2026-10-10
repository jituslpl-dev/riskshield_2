import numpy as np


def population_stability_index(reference, current, bins=10):
    """Score PSI over fixed [0,1] bins; drift is not performance degradation."""
    reference, current = np.asarray(reference), np.asarray(current)
    for values in [reference, current]:
        if len(values) < 100 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("At least 100 finite probabilities in [0,1] required per window.")
    edges = np.linspace(0, 1, bins + 1)
    a = np.histogram(reference, edges)[0].astype(float) + 0.5
    b = np.histogram(current, edges)[0].astype(float) + 0.5
    a /= a.sum()
    b /= b.sum()
    psi = float(np.sum((b - a) * np.log(b / a)))
    return {
        "psi": psi,
        "reference_count": len(reference),
        "current_count": len(current),
        "warning": bool(psi > 0.2),
        "note": "PSI threshold 0.2 is heuristic; investigate before acting.",
    }
