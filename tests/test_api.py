import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from riskshield.legacy_api import app, bundle
from riskshield.pipeline import FEATURES


def test_prediction_validation_and_shap_additivity():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 6)), columns=FEATURES)
    y = (X.amount > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    cal = LogisticRegression().fit(model.decision_function(X).reshape(-1, 1), y)
    b = {
        "model": model,
        "calibration": ("platt", cal),
        "threshold": 0.5,
        "features": FEATURES,
        "background": X.iloc[:20],
    }
    app.dependency_overrides.clear()
    import riskshield.legacy_api as api

    original = api.bundle
    api.bundle = lambda: b
    try:
        client = TestClient(app)
        sample = dict(amount=1, hour=2, foreign=0, account_age_days=1, prior_count_1h=0, prior_amount_1h=0)
        response = client.post("/explain", json=sample)
        assert response.status_code == 200
        result = response.json()
        raw = model.predict_proba(pd.DataFrame([sample])[FEATURES])[0, 1]
        assert abs(result["base_value"] + sum(result["contributions"].values()) - raw) < 1e-6
        assert 0 <= result["fraud_probability"] <= 1
        assert client.post("/predict", json={**sample, "amount": -1}).status_code == 422
    finally:
        api.bundle = original
