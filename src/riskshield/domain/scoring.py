"""Numerical adapters for the retained fitted calibration objects."""

import numpy as np


def calibrated(model, calibration, X):
    p = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    kind, object_ = calibration
    if kind == "platt":
        return object_.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    if kind == "isotonic":
        return object_.predict(p)
    raise ValueError("Unknown calibration method.")
