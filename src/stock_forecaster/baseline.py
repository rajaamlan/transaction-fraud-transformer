from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_forecaster.features import modeling_columns


@dataclass(frozen=True)
class BaselineRangeModel:
    """A simple benchmark forecaster based on average historical future return."""

    horizon_days: int
    expected_return: float
    mean_absolute_error: float

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        predictions = frame[["date", "ticker", "close"]].copy()
        predictions["expected_return"] = self.expected_return
        predictions["expected_close"] = predictions["close"] * (1.0 + self.expected_return)
        predictions["lower_close"] = predictions["expected_close"] * (1.0 - self.mean_absolute_error)
        predictions["upper_close"] = predictions["expected_close"] * (1.0 + self.mean_absolute_error)
        predictions["historical_error_pct"] = self.mean_absolute_error
        return predictions


def fit_baseline_range_model(training_frame: pd.DataFrame, horizon_days: int = 5) -> BaselineRangeModel:
    _, target_return, _ = modeling_columns(horizon_days)
    if target_return not in training_frame.columns:
        raise ValueError(f"Training frame does not contain target column: {target_return}")

    target = training_frame[target_return].dropna()
    if target.empty:
        raise ValueError("Training frame has no usable target values.")

    expected_return = float(target.mean())
    mean_absolute_error = float((target - expected_return).abs().mean())
    return BaselineRangeModel(
        horizon_days=horizon_days,
        expected_return=expected_return,
        mean_absolute_error=mean_absolute_error,
    )

