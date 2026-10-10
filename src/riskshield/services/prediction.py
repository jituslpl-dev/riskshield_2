import hashlib
import json
import math
import uuid
from dataclasses import asdict
import numpy as np
import pandas as pd
from riskshield.domain.policy import CostPolicy
from riskshield.infrastructure.registry import ModelRepository
from riskshield.domain.scoring import calibrated
from riskshield.domain.policy import decisions


class PredictionService:
    def __init__(self, repository: ModelRepository, audit, policy: CostPolicy):
        self.repository, self.audit, self.policy = repository, audit, policy

    @staticmethod
    def frame(rows, features):
        if not rows or len(rows) > 256:
            raise ValueError("Batch must contain 1 to 256 rows.")
        for row in rows:
            if set(row) != set(features):
                raise ValueError("Feature names must exactly match the model schema.")
            if any(
                isinstance(v, bool)
                or not isinstance(v, (float, int))
                or not math.isfinite(v)
                or abs(v) > 1e12
                for v in row.values()
            ):
                raise ValueError("Features must be finite bounded numbers.")
        return pd.DataFrame(rows)[features]

    def score(self, name, rows, key=None, policy_mode="expected_cost"):
        if policy_mode not in ["expected_cost", "legacy_threshold"]:
            raise ValueError("Unknown policy strategy.")
        bundle, manifest = self.repository.load(name)
        X = self.frame(rows, bundle["features"])
        digest = hashlib.sha256(
            json.dumps(
                {
                    "name": name,
                    "version": manifest["version"],
                    "rows": rows,
                    "policy_mode": policy_mode,
                    "policy": asdict(self.policy),
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        key = name + ":" + (key or str(uuid.uuid4()))
        with self.audit.lock:
            previous = self.audit.get(key, digest)
            if previous:
                return previous
            probabilities = calibrated(bundle["model"], bundle["calibration"], X)
            actions, costs = self.policy.evaluate(probabilities)
            if policy_mode == "legacy_threshold":
                actions = decisions(probabilities, bundle["threshold"], bundle.get("review_threshold"))
            response = {
                "model": name,
                "model_version": manifest["version"],
                "policy_mode": policy_mode,
                "predictions": [
                    {
                        "probability": float(p),
                        "decision": str(a),
                        "expected_costs": dict(zip(["approve", "review", "decline"], map(float, c))),
                    }
                    for p, a, c in zip(probabilities, actions, costs)
                ],
            }
            self.audit.save(key, digest, response)
            return response

    def explain(self, name, row):
        import shap

        bundle, manifest = self.repository.load(name)
        X = self.frame([row], bundle["features"])
        fn = lambda a: bundle["model"].predict_proba(pd.DataFrame(a, columns=bundle["features"]))[:, 1]
        explainer = shap.Explainer(fn, bundle["background"], seed=42)
        explanation = explainer(X, max_evals=max(2 * len(bundle["features"]) + 1, 128))
        return {
            "model_version": manifest["version"],
            "scale": "uncalibrated model probability",
            "raw_probability": float(fn(X)[0]),
            "calibrated_probability": float(calibrated(bundle["model"], bundle["calibration"], X)[0]),
            "base_value": float(explanation.base_values[0]),
            "contributions": dict(zip(bundle["features"], map(float, explanation.values[0]))),
        }
