"""Explicit, reproducible evaluation on public historical benchmark data."""

import argparse
import hashlib
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from .pipeline import train, calibrated


def explanations(output, prepared):
    import shap

    df, names, blocks, _ = prepared
    b = joblib.load(Path(output) / "model.joblib")
    rows = df.iloc[blocks[-1][:5]][names]
    predict_raw = lambda a: b["model"].predict_proba(pd.DataFrame(a, columns=names))[:, 1]
    explainer = shap.Explainer(predict_raw, b["background"], seed=42)
    explanation = explainer(rows, max_evals=2 * len(names) + 1)
    raw = predict_raw(rows)
    p = calibrated(b["model"], b["calibration"], rows)
    records = []
    for i in range(len(rows)):
        error = abs(float(explanation.base_values[i] + explanation.values[i].sum() - raw[i]))
        if error > 1e-6:
            raise AssertionError("SHAP additivity check failed.")
        records.append(
            {
                "test_row_position": i,
                "raw_probability": float(raw[i]),
                "calibrated_probability": float(p[i]),
                "base_value": float(explanation.base_values[i]),
                "additivity_error": error,
                "contributions": dict(zip(names, map(float, explanation.values[i]))),
            }
        )
    (Path(output) / "explanations.json").write_text(
        json.dumps(
            {
                "scale": "uncalibrated model probability",
                "method": "model-agnostic SHAP; minimum permutation budget, five held-out rows",
                "examples": records,
            },
            indent=2,
        )
    )


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def chronological_blocks(times):
    # Move boundaries to first event at the selected timestamp: ties stay together.
    n = len(times)
    cuts = [int(np.searchsorted(times, times[int(n * f)], side="left")) for f in [0.55, 0.70, 0.80, 0.90]]
    return np.split(np.arange(n), cuts)


def prepare_fraud(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    if "Time" not in df:
        raise ValueError("Dataset must retain Time for chronological evaluation.")
    initial = len(df)
    df = df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    df["label"] = df["Class"].astype(int)
    names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    blocks = chronological_blocks(df.Time.to_numpy())
    metadata = {
        "data": "real anonymized ULB/Worldline credit-card fraud benchmark",
        "source": "https://www.openml.org/search?type=data&id=1597",
        "sha256": file_hash(path),
        "split_protocol": "chronological by Time; simultaneous timestamps stay in one split",
        "duplicates_removed": initial - len(df),
        "features": names,
        "limitations": [
            "Only two days of historical 2013 data; not production validation.",
            "No account identifiers: account velocity features cannot be evaluated.",
            "V1-V28 are publisher-provided PCA features; their original preprocessing cannot be audited.",
            "SHAP values for PCA components are mathematical contributions, not human-readable reasons.",
        ],
        "split_time_ranges": {
            name: [float(df.Time.iloc[idx].min()), float(df.Time.iloc[idx].max())]
            for name, idx in zip(["train", "calibration", "selection", "policy", "test"], blocks)
        },
    }
    return df, names, blocks, metadata


def prepare_credit(path):
    df = pd.read_excel(path, header=1)
    initial = len(df)
    # Deduplicate modeling records before splitting; exclude ID and demographics.
    df = df.drop(columns=["ID"]).drop_duplicates().reset_index(drop=True)
    target = "default payment next month"
    df["label"] = df[target].astype(int)
    names = [
        column for column in df if column not in [target, "label", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    ]
    remaining = np.arange(len(df))
    blocks = []
    for fraction in [0.55, 0.15 / 0.45, 0.10 / 0.30, 0.10 / 0.20]:
        selected, remaining = train_test_split(
            remaining, train_size=fraction, stratify=df.label.iloc[remaining], random_state=42
        )
        blocks.append(selected)
    blocks.append(remaining)
    metadata = {
        "data": "real UCI Default of Credit Card Clients benchmark",
        "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. DOI:10.24432/C55S3H. CC BY 4.0.",
        "sha256": file_hash(path),
        "split_protocol": "five disjoint stratified random splits, seed 42; no application timestamps available",
        "duplicates_removed": initial - len(df),
        "features": names,
        "limitations": [
            "Historical Taiwanese credit-card snapshot; temporal generalization cannot be tested.",
            "Demographic columns excluded; this does not establish fairness.",
            "Features describe existing card accounts, not new loan applications.",
            "Illustrative perfect-review costs are not real loss estimates.",
        ],
    }
    return df, names, blocks, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fraud", "credit"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    prepared = prepare_fraud(args.input) if args.task == "fraud" else prepare_credit(args.input)
    output = f"artifacts/real_{args.task}"
    train(output=output, task=args.task, prepared=prepared)
    explanations(output, prepared)
