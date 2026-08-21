from __future__ import annotations

from fastapi import FastAPI

from fraud_model.agent import FraudAgent
from fraud_model.schemas import FraudScore, Transaction


app = FastAPI(title="Transaction Fraud Transformer Agent", version="0.1.0")
agent = FraudAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_status": agent.model_status}


@app.post("/score", response_model=FraudScore)
def score(transaction: Transaction) -> FraudScore:
    return agent.score(transaction)


@app.post("/score-batch", response_model=list[FraudScore])
def score_batch(transactions: list[Transaction]) -> list[FraudScore]:
    return [agent.score(transaction) for transaction in transactions]
