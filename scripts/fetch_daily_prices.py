from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_forecaster.data_sources import download_yfinance_prices
from stock_forecaster.storage import (
    initialize_database,
    read_prices_daily,
    table_counts,
    upsert_prices_daily,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily stock prices into SQLite.")
    parser.add_argument(
        "--config",
        default="configs/stock_data_architecture.json",
        help="Path to the stock data architecture config.",
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker override.")
    parser.add_argument("--start-date", default="2015-01-01", help="Inclusive start date.")
    parser.add_argument("--end-date", default=None, help="Exclusive end date. Defaults to today.")
    parser.add_argument("--source", default="yfinance", choices=["yfinance"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    database_path = Path(config["database_path"])
    tickers = args.tickers or config["starter_universe"]

    initialize_database(database_path)
    if args.source == "yfinance":
        prices = download_yfinance_prices(tickers, start_date=args.start_date, end_date=args.end_date)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    upsert_prices_daily(prices, database_path, source=args.source)
    counts = table_counts(database_path)
    sample = read_prices_daily(database_path, tickers=tickers[:2]).tail(10)

    print(f"Fetched {len(prices):,} daily price rows from {args.source}.")
    print(json.dumps(counts, indent=2))
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()

