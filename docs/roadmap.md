# Roadmap

This roadmap keeps the repo useful for learning while moving it toward a more
impressive public portfolio project.

## Now

- Keep the IEEE-CIS fraud-detection experiment reproducible.
- Maintain unit tests for preprocessing, submission validation, and stock
  feature generation.
- Build the stock forecaster in small, explainable steps.

## Next

- Add a stock data collection script for a single ticker.
- Save OHLCV data locally as CSV or Parquet.
- Create `notebooks/01_stock_forecaster_walkthrough.ipynb`.
- Add chronological train/validation splits for the stock forecasting lab.
- Compare the baseline range model against Random Forest and gradient boosting.

## Later

- Add XGBoost or LightGBM if dependency setup stays lightweight.
- Add a backtesting report with hit rate, error distribution, drawdown, and
  buy-and-hold comparison.
- Build a small dashboard for forecasts, ranges, and historical error.
- Add paper-trading simulation before any broker integration.
- Add portfolio risk rules, position limits, and human approval workflows.

## Not Yet

- No automated real-money trading.
- No options, margin, or leverage.
- No claims that the model predicts exact future prices.

