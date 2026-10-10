import os
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import calibrated, decisions

app = FastAPI(title="RiskShield", version="1.0.0")


class Transaction(BaseModel):
    amount: float = Field(ge=0, le=1e8)
    hour: int = Field(ge=0, le=23)
    foreign: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    prior_count_1h: int = Field(ge=0)
    prior_amount_1h: float = Field(ge=0, le=1e10)


class Loan(BaseModel):
    loan_amount: float = Field(gt=0, le=1e8)
    annual_income: float = Field(gt=0, le=1e9)
    debt_to_income: float = Field(ge=0, le=1)
    credit_history_years: int = Field(ge=0, le=100)
    prior_delinquencies: int = Field(ge=0, le=100)
    interest_rate: float = Field(ge=0, le=1)


@lru_cache
def credit_bundle():
    try:
        return joblib.load(os.getenv("RISKSHIELD_CREDIT_MODEL", "artifacts/credit/model.joblib"))
    except FileNotFoundError:
        raise HTTPException(503, "Credit model unavailable; train with --task credit.")


@app.post("/credit/predict")
def predict_credit(loan: Loan):
    b = credit_bundle()
    X = pd.DataFrame([loan.model_dump()])[b["features"]]
    p = float(calibrated(b["model"], b["calibration"], X)[0])
    return {
        "default_probability": p,
        "decision": str(decisions(np.array([p]), b["threshold"], b.get("review_threshold"))[0]),
        "decline_threshold": float(b["threshold"]),
        "review_threshold": float(b.get("review_threshold", b["threshold"] / 2)),
    }


@app.post("/credit/explain")
def explain_credit(loan: Loan):
    b = credit_bundle()
    X = pd.DataFrame([loan.model_dump()])[b["features"]]
    explainer = shap.Explainer(
        lambda a: b["model"].predict_proba(pd.DataFrame(a, columns=b["features"]))[:, 1], b["background"]
    )
    explanation = explainer(X)
    return {
        **predict_credit(loan),
        "explanation_scale": "uncalibrated model probability",
        "base_value": float(explanation.base_values[0]),
        "contributions": dict(zip(b["features"], map(float, explanation.values[0]))),
    }


@lru_cache
def bundle():
    try:
        return joblib.load(os.getenv("RISKSHIELD_MODEL", "artifacts/model.joblib"))
    except FileNotFoundError:
        raise HTTPException(503, "Model unavailable; run training first.")


@app.get("/health")
def health():
    bundle()
    return {"status": "ready"}


@app.post("/predict")
def predict(transaction: Transaction):
    b = bundle()
    X = pd.DataFrame([transaction.model_dump()])[b["features"]]
    p = float(calibrated(b["model"], b["calibration"], X)[0])
    return {
        "fraud_probability": p,
        "decision": str(decisions(np.array([p]), b["threshold"], b.get("review_threshold"))[0]),
        "decline_threshold": float(b["threshold"]),
        "review_threshold": float(b.get("review_threshold", b["threshold"] / 2)),
    }


@app.post("/explain")
def explain(transaction: Transaction):
    b = bundle()
    X = pd.DataFrame([transaction.model_dump()])[b["features"]]
    explainer = shap.Explainer(
        lambda a: b["model"].predict_proba(pd.DataFrame(a, columns=b["features"]))[:, 1], b["background"]
    )
    explanation = explainer(X)
    return {
        **predict(transaction),
        "explanation_scale": "uncalibrated model probability",
        "base_value": float(explanation.base_values[0]),
        "contributions": dict(zip(b["features"], map(float, explanation.values[0]))),
    }
