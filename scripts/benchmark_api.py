"""Local in-process latency sample; not a network or concurrency load test."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from riskshield.api import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    row = {
        "amount": 250,
        "hour": 2,
        "foreign": 1,
        "account_age_days": 15,
        "prior_count_1h": 3,
        "prior_amount_1h": 600,
    }
    times = []
    with TestClient(app) as client:
        for _ in range(5):
            client.post("/v1/models/fraud_demo/predict", json={"rows": [row]}).raise_for_status()
        for _ in range(args.requests):
            start = time.perf_counter()
            response = client.post("/v1/models/fraud_demo/predict", json={"rows": [row]})
            response.raise_for_status()
            times.append((time.perf_counter() - start) * 1000)
    report = {
        "method": "sequential in-process FastAPI TestClient, warm model; includes SQLite audit writes",
        "requests": len(times),
        "p50_ms": float(np.quantile(times, 0.5)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "note": "Not a production SLA, network benchmark, or concurrent load test.",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/local_latency.json").write_text(json.dumps(report, indent=2))
    print(report)
