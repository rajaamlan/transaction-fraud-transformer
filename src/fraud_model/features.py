from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_model.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


UNKNOWN_TOKEN = "<unknown>"


@dataclass
class FeaturePreprocessor:
    numeric_features: list[str]
    categorical_features: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    category_maps: dict[str, dict[str, int]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> "FeaturePreprocessor":
        numeric_features = numeric_features or NUMERIC_FEATURES
        categorical_features = categorical_features or CATEGORICAL_FEATURES

        numeric = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean().to_dict()
        stds = numeric.std(ddof=0).replace(0, 1.0).fillna(1.0).to_dict()

        category_maps: dict[str, dict[str, int]] = {}
        for feature in categorical_features:
            values = sorted(frame[feature].fillna(UNKNOWN_TOKEN).astype(str).unique())
            category_maps[feature] = {UNKNOWN_TOKEN: 0}
            category_maps[feature].update({value: idx + 1 for idx, value in enumerate(values)})

        return cls(
            numeric_features=list(numeric_features),
            categorical_features=list(categorical_features),
            numeric_mean={key: float(value) for key, value in means.items()},
            numeric_std={key: float(value) for key, value in stds.items()},
            category_maps=category_maps,
        )

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = frame[self.numeric_features].apply(pd.to_numeric, errors="coerce").copy()
        for feature in self.numeric_features:
            numeric[feature] = (
                numeric[feature].fillna(self.numeric_mean[feature]) - self.numeric_mean[feature]
            ) / self.numeric_std[feature]

        categorical_arrays = []
        for feature in self.categorical_features:
            mapping = self.category_maps[feature]
            encoded = (
                frame[feature]
                .fillna(UNKNOWN_TOKEN)
                .astype(str)
                .map(lambda value: mapping.get(value, 0))
                .astype(int)
            )
            categorical_arrays.append(encoded.to_numpy())

        categorical = np.vstack(categorical_arrays).T if categorical_arrays else np.empty((len(frame), 0))
        return numeric.to_numpy(dtype=np.float32), categorical.astype(np.int64)

    def category_cardinalities(self) -> list[int]:
        return [len(self.category_maps[feature]) for feature in self.categorical_features]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeaturePreprocessor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
