from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_forecaster.storage import initialize_database, table_counts, upsert_tickers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the stock research database.")
    parser.add_argument(
        "--config",
        default="configs/stock_data_architecture.json",
        help="Path to the stock data architecture config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    database_path = Path(config["database_path"])

    initialize_database(database_path)
    starter_universe = pd.DataFrame(
        [{"ticker": ticker, "asset_type": "stock", "is_active": 1} for ticker in config["starter_universe"]]
    )
    upsert_tickers(starter_universe, database_path)

    print(f"Initialized stock database: {database_path}")
    print(json.dumps(table_counts(database_path), indent=2))


if __name__ == "__main__":
    main()

