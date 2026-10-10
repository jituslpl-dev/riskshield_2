"""Chronological model comparison with separate calibration and policy windows."""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

FEATURES = ["amount", "hour", "foreign", "account_age_days", "prior_count_1h", "prior_amount_1h"]
CREDIT_FEATURES = [
    "loan_amount",
    "annual_income",
    "debt_to_income",
    "credit_history_years",
    "prior_delinquencies",
    "interest_rate",
]


def synthetic_credit(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
            "loan_amount": rng.lognormal(9.3, 0.5, n),
            "annual_income": rng.lognormal(11, 0.5, n),
            "debt_to_income": rng.beta(2, 5, n),
            "credit_history_years": rng.integers(0, 25, n),
            "prior_delinquencies": rng.poisson(0.3, n),
            "interest_rate": rng.uniform(0.04, 0.25, n),
        }
    )
    logit = (
        -5
        + 4 * df.debt_to_income
        + 0.6 * df.prior_delinquencies
        + 3 * df.interest_rate
        + 0.7 * (df.credit_history_years < 2)
    )
    df["label"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def synthetic(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
            "account_id": rng.integers(0, 500, n),
            "amount": rng.lognormal(3.5, 1.2, n),
            "foreign": rng.binomial(1, 0.08, n),
            "account_age_days": rng.integers(1, 2000, n),
        }
    )
    x = features(df)
    logit = (
        -6
        + 1.8 * x.foreign
        + 0.004 * x.amount.clip(upper=1200)
        + 1.2 * (x.account_age_days < 60)
        + 0.7 * x.prior_count_1h
    )
    df["label"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def features(df):
    """Only strictly earlier account events contribute; timestamp ties are excluded."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["hour"] = df.timestamp.dt.hour
    df["prior_count_1h"] = 0.0
    df["prior_amount_1h"] = 0.0
    for _, group in df.groupby("account_id"):
        times = group.timestamp.astype("int64").to_numpy()
        amounts = group.amount.to_numpy()
        start = np.searchsorted(times, times - 3600 * 10**9, side="left")
        end = np.searchsorted(times, times, side="left")
        total = np.r_[0, amounts.cumsum()]
        df.loc[group.index, "prior_count_1h"] = end - start
        df.loc[group.index, "prior_amount_1h"] = total[end] - total[start]
    return df


from riskshield.domain.scoring import calibrated
from riskshield.domain.policy import decisions


def train(output="artifacts", n=20000, task="fraud", prepared=None):
    metadata = {
        "data": "synthetic demonstration; not validated for real-world decisions",
        "split_protocol": "chronological",
    }
    if prepared is not None:
        df, feature_names, blocks, metadata = prepared
        n = len(df)
    else:
        df = features(synthetic(n)) if task == "fraud" else synthetic_credit(n)
        feature_names = FEATURES if task == "fraud" else CREDIT_FEATURES
    # Five independent chronological windows: train, calibration, selection, policy, test.
    if prepared is None:
        blocks = np.split(np.arange(n), [int(n * 0.55), int(n * 0.70), int(n * 0.80), int(n * 0.90)])
    train_i, cal_i, select_i, policy_i, test_i = blocks
    X, y = df[feature_names], df.label
    for idx in blocks:
        if y.iloc[idx].nunique() != 2:
            raise ValueError("Each chronological window must contain both classes; increase n.")
    ratio = (y.iloc[train_i] == 0).sum() / y.iloc[train_i].sum()
    from riskshield.training.factory import ClassifierFactory

    models = ClassifierFactory.create(ratio)
    candidates = []
    for name, model in models.items():
        model.fit(X.iloc[train_i], y.iloc[train_i])
        raw = np.clip(model.predict_proba(X.iloc[cal_i])[:, 1], 1e-7, 1 - 1e-7)
        calibrators = [
            ("platt", LogisticRegression(C=1e6).fit(np.log(raw / (1 - raw)).reshape(-1, 1), y.iloc[cal_i])),
            ("isotonic", IsotonicRegression(out_of_bounds="clip").fit(raw, y.iloc[cal_i])),
        ]
        for calibration in calibrators:
            p = calibrated(model, calibration, X.iloc[select_i])
            candidates.append((brier_score_loss(y.iloc[select_i], p), name, model, calibration))
    best = min(candidates, key=lambda c: c[0])
    _, name, model, calibration = best
    p = calibrated(model, calibration, X.iloc[policy_i])

    # Illustrative fixed costs: false decline 5, missed fraud 100, perfect review 1.
    def cost(prob, truth, threshold, review_threshold):
        action = decisions(prob, threshold, review_threshold)
        return float(
            np.mean(
                np.where(action == "review", 1, np.where(action == "decline", 5 * (1 - truth), 100 * truth))
            )
        )

    pairs = [
        (low, high)
        for low in np.r_[0, np.linspace(0.001, 0.1, 50)]
        for high in np.r_[np.linspace(0.05, 0.99, 50), 1.01]
        if low <= high
    ]
    review_threshold, threshold = min(
        pairs, key=lambda pair: cost(p, y.iloc[policy_i].to_numpy(), pair[1], pair[0])
    )
    results = []
    for score, candidate_name, candidate_model, candidate_cal in candidates:
        prob = calibrated(candidate_model, candidate_cal, X.iloc[test_i])
        results.append(
            {
                "model": candidate_name,
                "calibration": candidate_cal[0],
                "selection_brier": score,
                "test_pr_auc": average_precision_score(y.iloc[test_i], prob),
                "test_brier": brier_score_loss(y.iloc[test_i], prob),
            }
        )
    prob = calibrated(model, calibration, X.iloc[test_i])
    actions = decisions(prob, threshold, review_threshold)
    report = {
        **metadata,
        "task": task,
        "rows": n,
        "event_rate": float(y.mean()),
        "selected_model": name,
        "calibration": calibration[0],
        "decline_threshold": float(threshold),
        "review_threshold": float(review_threshold),
        "test_cost_per_transaction": cost(prob, y.iloc[test_i].to_numpy(), threshold, review_threshold),
        "review_all_test_cost": 1.0,
        "approve_all_test_cost": float(y.iloc[test_i].mean() * 100),
        "test_action_counts": {
            action: int((actions == action).sum()) for action in ["approve", "review", "decline"]
        },
        "test_event_rate": float(y.iloc[test_i].mean()),
        "split_positive_counts": dict(
            zip(
                ["train", "calibration", "selection", "policy", "test"],
                [int(y.iloc[idx].sum()) for idx in blocks],
            )
        ),
        "split_rows": dict(zip(["train", "calibration", "selection", "policy", "test"], map(len, blocks))),
        "models": results,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "calibration": calibration,
            "threshold": threshold,
            "review_threshold": review_threshold,
            "features": feature_names,
            "background": X.iloc[train_i].sample(80, random_state=42),
        },
        output / "model.joblib",
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--task", choices=["fraud", "credit"], default="fraud")
    args = parser.parse_args()
    train(args.output or ("artifacts" if args.task == "fraud" else "artifacts/credit"), args.rows, args.task)
