# Stock Price Range Forecaster Strategy

This project should become a forecasting and portfolio research assistant, not a
magic stock-price machine. The first useful version predicts a future price range,
shows its historical error, and later turns those forecasts into recommendations
only after backtesting.

## Product Goal

Build a tool that answers:

> Given the information available at market close today, what is the likely price
> range for this stock over the next few trading days?

The tool should produce outputs like:

| Field | Example |
| --- | --- |
| Ticker | AAPL |
| Horizon | 5 trading days |
| Expected close | 234.10 |
| Predicted range | 226.80 to 241.40 |
| Historical error | +/- 3.2% |
| Signal | mildly bullish |

This is educational and research-focused. It is not financial advice and should
not automate real trades until it has gone through careful backtesting, paper
trading, risk controls, and human approval.

## Step-By-Step Build Plan

1. Create a leakage-safe tabular dataset from historical OHLCV prices.
2. Train a simple baseline model and measure historical error.
3. Train a stronger tabular model such as Random Forest, XGBoost, or LightGBM.
4. Backtest predictions across different market periods.
5. Build a small dashboard that shows forecasts, ranges, and model errors.
6. Add a portfolio recommendation layer.
7. Add paper trading.
8. Only later consider broker execution with strict limits and human approval.

## First Prediction Target

Start with a 5-trading-day horizon:

```text
future_return_5d = close_price_5_trading_days_from_now / close_today - 1
future_close_5d = close_price_5_trading_days_from_now
```

The model should use only current and historical features, such as:

- 1-day return
- 5-day return
- 20-day return
- 5-day volatility
- 20-day volatility
- close versus moving averages
- volume change

## Modeling Rule

Start with a baseline before any advanced model:

```text
expected_future_return = recent average horizon return
prediction_range = expected price +/- historical absolute forecast error
```

Then compare machine-learning models against that baseline. A model is only useful
if it improves backtest behavior after transaction costs, risk limits, and bad
market periods are considered.

