from __future__ import annotations

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    merchant_risk: float = Field(ge=0, le=1)
    customer_age_days: int = Field(ge=0)
    transactions_last_24h: int = Field(ge=0)
    country: str
    merchant_category: str
    payment_method: str


class FraudScore(BaseModel):
    risk_score: float
    decision: str
    reason_codes: list[str]
    model_status: str
