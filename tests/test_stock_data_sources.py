import unittest

import pandas as pd

from stock_forecaster.data_sources import normalize_yfinance_prices


class StockDataSourceTests(unittest.TestCase):
    def test_normalize_yfinance_prices_handles_multi_ticker_download(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        columns = pd.MultiIndex.from_product(
            [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
        downloaded = pd.DataFrame(
            [
                [100, 101, 99, 100.5, 100.2, 1000, 200, 201, 199, 200.5, 200.2, 2000],
                [101, 102, 100, 101.5, 101.2, 1100, 201, 202, 200, 201.5, 201.2, 2100],
            ],
            index=dates,
            columns=columns,
        )

        prices = normalize_yfinance_prices(downloaded, ["aapl", "msft"])

        self.assertEqual(len(prices), 4)
        self.assertEqual(
            list(prices.columns),
            ["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"],
        )
        self.assertEqual(prices["ticker"].unique().tolist(), ["AAPL", "MSFT"])
        self.assertEqual(int(prices.loc[0, "volume"]), 1000)


if __name__ == "__main__":
    unittest.main()

