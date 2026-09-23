# Stock Data Architecture

The stock project should treat data as the product foundation. The first version
uses local SQLite for structured tables and leaves room to add Parquet archives
or cloud storage later.

## Storage Layers

| Layer | Location | Purpose |
| --- | --- | --- |
| Code and docs | GitHub | Versioned source, tests, schema docs |
| Local database | `data/market_data.sqlite` | Queryable research database |
| Raw archive | `data/raw/` | Downloaded source files, not committed |
| Processed archive | `data/processed/` | Feature exports, not committed |
| Future cloud archive | External 5TB storage | Large Parquet history and backups |

GitHub should hold the recipe, not the whole pantry. Raw data, processed data,
SQLite files, model checkpoints, credentials, and broker exports should stay out
of git.

## Core Tables

### `tickers`

One row per tradable instrument.

| Column | Meaning |
| --- | --- |
| `ticker` | Symbol, primary key |
| `name` | Company or fund name |
| `exchange` | Listing venue |
| `sector` | Sector, if available |
| `industry` | Industry, if available |
| `asset_type` | Stock, ETF, index, crypto, etc. |
| `is_active` | Whether to include in active research |
| `created_at` | Insert timestamp |
| `updated_at` | Last update timestamp |

### `prices_daily`

One row per ticker per trading date.

| Column | Meaning |
| --- | --- |
| `date` | Trading date |
| `ticker` | Symbol |
| `open` | Daily open |
| `high` | Daily high |
| `low` | Daily low |
| `close` | Daily close |
| `adjusted_close` | Split/dividend adjusted close |
| `volume` | Daily volume |
| `source` | Data vendor |
| `created_at` | Insert timestamp |

Primary key: `(date, ticker)`.

### `features_daily`

One row per ticker per trading date after feature generation. These columns must
only use information known at or before that date.

Examples:

- `return_1d`
- `return_5d`
- `return_20d`
- `volatility_5d`
- `volatility_20d`
- `close_to_ma_5`
- `close_to_ma_20`
- `volume_to_ma_20`
- `intraday_range`
- `close_position`

### `targets_daily`

One row per ticker per trading date for future-looking labels. These are allowed
for training and evaluation, but must never be model input features.

Examples:

- `horizon_days`
- `future_close`
- `future_return`

Primary key: `(date, ticker, horizon_days)`.

## Modeling Principle

The data is time-series data, but the first model should be tabular ML with
time-aware validation:

```text
raw prices -> features known today -> future target -> chronological split -> model
```

This lets one model learn across many tickers while respecting the calendar.

## First Milestone

1. Create the local SQLite database.
2. Insert a small ticker universe.
3. Load daily OHLCV data.
4. Generate features and targets.
5. Train a baseline model.
6. Backtest on later dates only.

