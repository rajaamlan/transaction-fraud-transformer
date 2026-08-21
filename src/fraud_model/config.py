from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NUMERIC_FEATURES = [
    "amount",
    "hour",
    "merchant_risk",
    "customer_age_days",
    "transactions_last_24h",
]

CATEGORICAL_FEATURES = [
    "country",
    "merchant_category",
    "payment_method",
]


@dataclass(frozen=True)
class ArtifactPaths:
    artifact_dir: Path

    @property
    def preprocessor(self) -> Path:
        return self.artifact_dir / "preprocessor.json"

    @property
    def model(self) -> Path:
        return self.artifact_dir / "model.pt"

    @property
    def metrics(self) -> Path:
        return self.artifact_dir / "metrics.json"
