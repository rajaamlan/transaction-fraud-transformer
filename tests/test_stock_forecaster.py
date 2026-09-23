import unittest

import numpy as np
import pandas as pd

from stock_forecaster.baseline import fit_baseline_range_model
from stock_forecaster.features import (
    FeatureConfig,
    build_price_features,
    drop_unusable_training_rows,
    modeling_columns,
)


def make_price_frame(days=40):
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    rows = []
    for offset, ticker in enumerate(["AAA", "BBB"]):
        for index, date in enumerate(dates):
            close = 100 + offset * 20 + index * 0.5
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.2,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + index * 1_000,
                }
            )
    return pd.DataFrame(rows)


class StockForecasterTests(unittest.TestCase):
    def test_build_price_features_creates_future_target_without_crossing_tickers(self):
        frame = make_price_frame()

        features = build_price_features(frame, FeatureConfig(horizon_days=5))
        _, target_return, target_close = modeling_columns(5)

        first_aaa = features[features["ticker"] == "AAA"].iloc[0]
        last_aaa = features[features["ticker"] == "AAA"].iloc[-1]

        self.assertIn(target_return, features.columns)
        self.assertIn(target_close, features.columns)
        self.assertTrue(np.isclose(first_aaa[target_close], 102.5))
        self.assertTrue(pd.isna(last_aaa[target_close]))

    def test_drop_unusable_training_rows_removes_warmup_and_futureless_rows(self):
        features = build_price_features(make_price_frame(), FeatureConfig(horizon_days=5))

        training = drop_unusable_training_rows(features, horizon_days=5)

        self.assertFalse(training.empty)
        self.assertTrue(training["return_20d"].notna().all())
        self.assertTrue(training["future_return_5d"].notna().all())

    def test_baseline_predicts_range_columns(self):
        features = build_price_features(make_price_frame(), FeatureConfig(horizon_days=5))
        training = drop_unusable_training_rows(features, horizon_days=5)

        model = fit_baseline_range_model(training, horizon_days=5)
        predictions = model.predict_frame(training.tail(3))

        self.assertEqual(
            list(predictions.columns),
            [
                "date",
                "ticker",
                "close",
                "expected_return",
                "expected_close",
                "lower_close",
                "upper_close",
                "historical_error_pct",
            ],
        )
        self.assertTrue((predictions["lower_close"] <= predictions["expected_close"]).all())
        self.assertTrue((predictions["upper_close"] >= predictions["expected_close"]).all())


if __name__ == "__main__":
    unittest.main()
