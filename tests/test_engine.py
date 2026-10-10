import json
import numpy as np
import pandas as pd
import pytest
import joblib
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient
from riskshield.api import Settings, create_app
from riskshield.domain.policy import CostPolicy
from riskshield.infrastructure.audit import AuditStore
from riskshield.infrastructure.registry import FileModelRegistry
from riskshield.services.prediction import PredictionService
from riskshield.monitoring import population_stability_index


@pytest.fixture
def engine(tmp_path):
    X = pd.DataFrame({"x": np.linspace(-2, 2, 100)})
    y = (X.x > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    calibration = LogisticRegression().fit(model.decision_function(X).reshape(-1, 1), y)
    source = tmp_path / "source"
    source.mkdir()
    joblib.dump(
        {
            "model": model,
            "calibration": ("platt", calibration),
            "features": ["x"],
            "threshold": 0.5,
            "review_threshold": 0.1,
            "background": X.iloc[::5],
        },
        source / "model.joblib",
    )
    (source / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": "logistic",
                "calibration": "platt",
                "data": "unit fixture",
                "task": "fraud",
                "event_rate": 0.5,
            }
        )
    )
    registry = FileModelRegistry(tmp_path / "registry")
    registry.register("fixture", "v1", source)
    registry.promote("fixture", "v1")
    audit = AuditStore(tmp_path / "audit.sqlite")
    yield registry, audit, source
    audit.close()


def test_expected_cost_policy_handles_extremes_and_imperfect_review():
    policy = CostPolicy()
    action, cost = policy.evaluate([0, 0.05, 1])
    assert action.tolist() == ["approve", "review", "decline"]
    assert cost[1, 1] > policy.review_cost
    with pytest.raises(ValueError):
        CostPolicy(review_sensitivity=2)
    with pytest.raises(ValueError):
        policy.evaluate([float("nan")])


def test_registry_immutability_rollback_and_corruption(engine):
    registry, _, source = engine
    with pytest.raises(FileExistsError):
        registry.register("fixture", "v1", source)
    registry.register("fixture", "v2", source)
    registry.promote("fixture", "v2")
    assert registry.load("fixture")[1]["version"] == "v2"
    registry.promote("fixture", "v1")
    assert registry.load("fixture")[1]["version"] == "v1"
    with pytest.raises(ValueError):
        registry.load("../fixture")
    file = registry.root / "fixture/v2/model.joblib"
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        registry.promote("fixture", "v2")


def test_batch_idempotency_and_concurrent_retries(engine):
    registry, audit, _ = engine
    service = PredictionService(registry, audit, CostPolicy())
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: service.score("fixture", [{"x": 1.0}], key="same"), range(8)))
    assert all(response == responses[0] for response in responses)
    assert audit.connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    with pytest.raises(ValueError, match="Idempotency"):
        service.score("fixture", [{"x": 2.0}], key="same")
    with pytest.raises(ValueError):
        service.score("fixture", [{"x": float("inf")}])
    with pytest.raises(ValueError):
        service.score("fixture", [{"unexpected": 1.0}])
    assert len(service.score("fixture", [{"x": 1.0}, {"x": -1.0}])["predictions"]) == 2


def test_authenticated_api_and_explanation_additivity(engine, tmp_path):
    registry, audit, _ = engine
    app = create_app(Settings(audit_path=str(tmp_path / "unused"), api_key="secret"), registry, audit)
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").status_code == 401
        headers = {"X-API-Key": "secret", "Idempotency-Key": "retry"}
        body = {"rows": [{"x": 1.0}]}
        response = client.post("/v1/models/fixture/predict", json=body, headers=headers)
        assert response.status_code == 200 and "X-Request-ID" in response.headers
        assert client.post("/v1/models/fixture/predict", json=body, headers=headers).json() == response.json()
        assert (
            client.post(
                "/v1/models/fixture/predict", json={"rows": [{"x": 2.0}]}, headers=headers
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/models/fixture/predict", json={"rows": [{"x": "1"}]}, headers=headers
            ).status_code
            == 422
        )
        explanation = client.post(
            "/v1/models/fixture/explain", json={"features": {"x": 1.0}}, headers=headers
        ).json()
        assert (
            abs(
                explanation["base_value"]
                + sum(explanation["contributions"].values())
                - explanation["raw_probability"]
            )
            < 1e-6
        )
        assert "riskshield_http_requests_total" in client.get("/metrics", headers=headers).text


def test_production_fails_without_api_key(tmp_path):
    with pytest.raises(ValueError, match="Production"):
        create_app(Settings(environment="production", audit_path=str(tmp_path / "audit")))


def test_missing_model_and_unknown_policy_status(engine):
    registry, audit, _ = engine
    with TestClient(create_app(Settings(), registry, audit)) as client:
        assert client.post("/v1/models/missing/predict", json={"rows": [{"x": 1}]}).status_code == 503
        assert (
            client.post(
                "/v1/models/fixture/predict", json={"rows": [{"x": 1}], "policy_mode": "invalid"}
            ).status_code
            == 422
        )


def test_drift_detects_score_shift():
    baseline = np.linspace(0, 0.1, 200)
    assert population_stability_index(baseline, baseline)["psi"] == 0
    assert population_stability_index(baseline, np.linspace(0.8, 1, 200))["warning"]
