"""Demo request schemas; benchmark schemas come from registered manifests."""

from pydantic import BaseModel, Field, ConfigDict


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    amount: float = Field(ge=0, le=1e8)
    hour: int = Field(ge=0, le=23)
    foreign: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    prior_count_1h: int = Field(ge=0)
    prior_amount_1h: float = Field(ge=0, le=1e10)


class Loan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    loan_amount: float = Field(gt=0, le=1e8)
    annual_income: float = Field(gt=0, le=1e9)
    debt_to_income: float = Field(ge=0, le=1)
    credit_history_years: int = Field(ge=0, le=100)
    prior_delinquencies: int = Field(ge=0, le=100)
    interest_rate: float = Field(ge=0, le=1)
