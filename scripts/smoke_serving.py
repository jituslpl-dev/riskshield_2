"""Exercise all locally promoted model schemas through the public API."""

import json
from pathlib import Path
from fastapi.testclient import TestClient
from riskshield.api import app
from riskshield.infrastructure.registry import FileModelRegistry

if __name__ == "__main__":
    registry = FileModelRegistry("registry")
    results = []
    with TestClient(app) as client:
        client.get("/ready").raise_for_status()
        for name in registry.names():
            bundle, _ = registry.load(name)
            row = {k: float(v) for k, v in bundle["background"].iloc[0].to_dict().items()}
            response = client.post(f"/v1/models/{name}/predict", json={"rows": [row, row]})
            response.raise_for_status()
            explanation = client.post(f"/v1/models/{name}/explain", json={"features": row})
            explanation.raise_for_status()
            value = explanation.json()
            error = abs(value["base_value"] + sum(value["contributions"].values()) - value["raw_probability"])
            assert error < 1e-6
            results.append(
                {
                    "model": name,
                    "batch_status": response.status_code,
                    "explanation_status": explanation.status_code,
                    "additivity_error": error,
                }
            )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/serving_smoke.json").write_text(json.dumps(results, indent=2))
    print(results)
