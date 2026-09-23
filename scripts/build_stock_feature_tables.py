from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_forecaster.features import FeatureConfig, build_price_features
from stock_forecaster.storage import (
    read_modeling_frame,
    read_prices_daily,
    table_counts,
    upsert_features_daily,
    upsert_targets_daily,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stock feature and target tables.")
    parser.add_argument(
        "--config",
        default="configs/stock_data_architecture.json",
        help="Path to the stock data architecture config.",
    )
    parser.add_argument("--horizon-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    database_path = Path(config["database_path"])
    horizon_days = args.horizon_days or int(config["default_horizon_days"])

    prices = read_prices_daily(database_path)
    if prices.empty:
        raise ValueError("No prices found. Run scripts/fetch_daily_prices.py first.")

    feature_frame = build_price_features(prices, FeatureConfig(horizon_days=horizon_days))
    upsert_features_daily(feature_frame, database_path, horizon_days=horizon_days)
    upsert_targets_daily(feature_frame, database_path, horizon_days=horizon_days)

    modeling_frame = read_modeling_frame(database_path, horizon_days=horizon_days)
    print(json.dumps(table_counts(database_path), indent=2))
    print(f"Modeling rows for {horizon_days} day horizon: {len(modeling_frame):,}")
    print(modeling_frame.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()

