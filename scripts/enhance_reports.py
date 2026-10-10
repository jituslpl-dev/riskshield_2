"""Compute uncertainty and policy diagnostics using frozen v1 benchmark models."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from riskshield.domain.policy import CostPolicy
from riskshield.pipeline import calibrated
from riskshield.real_data import prepare_fraud, prepare_credit
from riskshield.training.evaluation import metrics, bootstrap_intervals
from riskshield.monitoring import population_stability_index


def evaluate(task, dataset, artifact):
    prepared = prepare_fraud(dataset) if task == "fraud" else prepare_credit(dataset)
    df, names, blocks, metadata = prepared
    model = joblib.load(Path(artifact) / "model.joblib")
    X = df.iloc[blocks[-1]][names]
    y = df.label.iloc[blocks[-1]].to_numpy()
    p = calibrated(model["model"], model["calibration"], X)
    reference = calibrated(model["model"], model["calibration"], df.iloc[blocks[-2]][names])
    policy = CostPolicy()
    actions, _ = policy.evaluate(p)
    costs = np.where(
        actions == "approve",
        y * policy.missed_event_cost,
        np.where(
            actions == "decline",
            (1 - y) * policy.false_decline_cost,
            policy.review_cost
            + y * (1 - policy.review_sensitivity) * policy.missed_event_cost
            + (1 - y) * (1 - policy.review_specificity) * policy.false_decline_cost,
        ),
    )
    report = {
        **metadata,
        "task": task,
        "test_metrics": metrics(y, p),
        "confidence_intervals": bootstrap_intervals(y, p),
        "policy": {
            "method": "expected costs; fixed before test evaluation, imperfect review assumed",
            "test_cost_per_record": float(costs.mean()),
            "action_counts": {a: int((actions == a).sum()) for a in ["approve", "review", "decline"]},
            "review_all_cost": float(
                policy.review_cost
                + y.mean() * (1 - policy.review_sensitivity) * policy.missed_event_cost
                + (1 - y.mean()) * (1 - policy.review_specificity) * policy.false_decline_cost
            ),
        },
        "score_shift_policy_to_test": population_stability_index(reference, p),
        "note": "Frozen original benchmark models; additional diagnostics do not refit or select on test data.",
    }
    output = Path("reports") / f"{task}_enhanced.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test_metrics"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fraud", "credit"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    evaluate(args.task, args.dataset, args.artifact)
