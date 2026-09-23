import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_forecaster.features import FeatureConfig, build_price_features
from stock_forecaster.storage import (
    initialize_database,
    read_modeling_frame,
    read_prices_daily,
    table_counts,
    upsert_features_daily,
    upsert_prices_daily,
    upsert_targets_daily,
    upsert_tickers,
)


class StockStorageTests(unittest.TestCase):
    def test_database_stores_tickers_and_prices_for_feature_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "market_data.sqlite"
            initialize_database(database_path)

            tickers = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "name": "Apple Inc.",
                        "exchange": "NASDAQ",
                        "sector": "Technology",
                        "industry": "Consumer Electronics",
                    }
                ]
            )
            prices = pd.DataFrame(
                [
                    {
                        "date": date,
                        "ticker": "AAPL",
                        "open": 100 + index,
                        "high": 101 + index,
                        "low": 99 + index,
                        "close": 100.5 + index,
                        "adjusted_close": 100.5 + index,
                        "volume": 1_000_000 + index,
                    }
                    for index, date in enumerate(pd.date_range("2024-01-01", periods=30, freq="B"))
                ]
            )

            upsert_tickers(tickers, database_path)
            upsert_prices_daily(prices, database_path, source="unit_test")

            loaded = read_prices_daily(database_path, tickers=["aapl"])
            features = build_price_features(loaded, FeatureConfig(horizon_days=5))
            upsert_features_daily(features, database_path, horizon_days=5)
            upsert_targets_daily(features, database_path, horizon_days=5)
            modeling_frame = read_modeling_frame(database_path, horizon_days=5)
            counts = table_counts(database_path)

            self.assertEqual(counts["tickers"], 1)
            self.assertEqual(counts["prices_daily"], 30)
            self.assertEqual(counts["features_daily"], 10)
            self.assertEqual(counts["targets_daily"], 25)
            self.assertEqual(len(loaded), 30)
            self.assertEqual(len(modeling_frame), 5)
            self.assertIn("future_return_5d", features.columns)
            self.assertIn("future_return", modeling_frame.columns)
            self.assertEqual(loaded["ticker"].unique().tolist(), ["AAPL"])


if __name__ == "__main__":
    unittest.main()
