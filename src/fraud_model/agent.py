from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from fraud_model.config import ArtifactPaths
from fraud_model.features import FeaturePreprocessor
from fraud_model.model import TransactionTransformer
from fraud_model.schemas import FraudScore, Transaction


class FraudAgent:
    def __init__(self, artifact_dir: str = "artifacts", threshold: float = 0.5) -> None:
        self.paths = ArtifactPaths(Path(artifact_dir))
        self.threshold = threshold
        self.preprocessor: FeaturePreprocessor | None = None
        self.model: TransactionTransformer | None = None
        self.model_status = "heuristic_fallback"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not self.paths.model.exists() or not self.paths.preprocessor.exists():
            return

        checkpoint = torch.load(self.paths.model, map_location="cpu")
        self.preprocessor = FeaturePreprocessor.load(self.paths.preprocessor)
        self.model = TransactionTransformer(
            num_numeric_features=checkpoint["num_numeric_features"],
            category_cardinalities=checkpoint["category_cardinalities"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.model_status = "trained_transformer"

    def score(self, transaction: Transaction) -> FraudScore:
        if self.model and self.preprocessor:
            risk_score = self._score_with_model(transaction)
        else:
            risk_score = self._score_with_heuristics(transaction)

        reason_codes = self._reason_codes(transaction, risk_score)
        decision = "review" if risk_score >= self.threshold else "approve"
        return FraudScore(
            risk_score=round(float(risk_score), 4),
            decision=decision,
            reason_codes=reason_codes,
            model_status=self.model_status,
        )

    def _score_with_model(self, transaction: Transaction) -> float:
        frame = pd.DataFrame([transaction.model_dump()])
        numeric, categorical = self.preprocessor.transform(frame)
        with torch.no_grad():
            logits = self.model(
                torch.tensor(numeric, dtype=torch.float32),
                torch.tensor(categorical, dtype=torch.long),
            )
            return float(torch.sigmoid(logits).item())

    def _score_with_heuristics(self, transaction: Transaction) -> float:
        score = 0.05
        score += min(transaction.amount / 2000, 0.35)
        score += transaction.merchant_risk * 0.35
        score += min(transaction.transactions_last_24h / 20, 0.20)
        if transaction.hour < 5:
            score += 0.10
        if transaction.customer_age_days < 30:
            score += 0.12
        return min(score, 0.99)

    def _reason_codes(self, transaction: Transaction, risk_score: float) -> list[str]:
        reasons: list[str] = []
        if transaction.amount >= 500:
            reasons.append("high_amount")
        if transaction.merchant_risk >= 0.7:
            reasons.append("high_merchant_risk")
        if transaction.transactions_last_24h >= 6:
            reasons.append("velocity_spike")
        if transaction.hour < 5:
            reasons.append("unusual_hour")
        if transaction.customer_age_days < 30:
            reasons.append("new_customer")
        if not reasons and risk_score >= self.threshold:
            reasons.append("model_pattern_match")
        return reasons or ["low_risk_profile"]
