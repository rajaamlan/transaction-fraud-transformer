from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for leakage-safe price feature generation."""

    horizon_days: int = 5


def validate_price_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")


def build_price_features(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Build tabular stock features and future-return targets.

    Each row represents what would have been known at the close of that date.
    Future columns are targets only; they must not be used as model inputs.
    """

    config = config or FeatureConfig()
    validate_price_frame(frame)

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = data.groupby("ticker", group_keys=False)

    data["return_1d"] = grouped["close"].pct_change(1)
    data["return_5d"] = grouped["close"].pct_change(5)
    data["return_20d"] = grouped["close"].pct_change(20)

    data["volatility_5d"] = grouped["return_1d"].rolling(5).std().reset_index(level=0, drop=True)
    data["volatility_20d"] = grouped["return_1d"].rolling(20).std().reset_index(level=0, drop=True)

    close_ma_5 = grouped["close"].rolling(5).mean().reset_index(level=0, drop=True)
    close_ma_20 = grouped["close"].rolling(20).mean().reset_index(level=0, drop=True)
    volume_ma_20 = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)

    data["close_to_ma_5"] = data["close"] / close_ma_5 - 1.0
    data["close_to_ma_20"] = data["close"] / close_ma_20 - 1.0
    data["volume_to_ma_20"] = data["volume"] / volume_ma_20 - 1.0
    data["intraday_range"] = data["high"] / data["low"] - 1.0
    data["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"])
    data["close_position"] = data["close_position"].replace([np.inf, -np.inf], np.nan)

    future_close = grouped["close"].shift(-config.horizon_days)
    data[f"future_close_{config.horizon_days}d"] = future_close
    data[f"future_return_{config.horizon_days}d"] = future_close / data["close"] - 1.0

    return data


def modeling_columns(horizon_days: int = 5) -> tuple[list[str], str, str]:
    feature_columns = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_5d",
        "volatility_20d",
        "close_to_ma_5",
        "close_to_ma_20",
        "volume_to_ma_20",
        "intraday_range",
        "close_position",
    ]
    target_return = f"future_return_{horizon_days}d"
    target_close = f"future_close_{horizon_days}d"
    return feature_columns, target_return, target_close


def drop_unusable_training_rows(feature_frame: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    feature_columns, target_return, target_close = modeling_columns(horizon_days)
    needed = feature_columns + [target_return, target_close]
    return feature_frame.dropna(subset=needed).reset_index(drop=True)

