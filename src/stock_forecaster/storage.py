from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from stock_forecaster.features import modeling_columns


DEFAULT_DATABASE_PATH = Path("data/market_data.sqlite")


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS tickers (
        ticker TEXT PRIMARY KEY,
        name TEXT,
        exchange TEXT,
        sector TEXT,
        industry TEXT,
        asset_type TEXT NOT NULL DEFAULT 'stock',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prices_daily (
        date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        adjusted_close REAL,
        volume INTEGER NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (date, ticker),
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS features_daily (
        date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        return_1d REAL,
        return_5d REAL,
        return_20d REAL,
        volatility_5d REAL,
        volatility_20d REAL,
        close_to_ma_5 REAL,
        close_to_ma_20 REAL,
        volume_to_ma_20 REAL,
        intraday_range REAL,
        close_position REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (date, ticker),
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS targets_daily (
        date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        horizon_days INTEGER NOT NULL,
        future_close REAL NOT NULL,
        future_return REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (date, ticker, horizon_days),
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_prices_daily_ticker_date ON prices_daily(ticker, date)",
    "CREATE INDEX IF NOT EXISTS idx_features_daily_ticker_date ON features_daily(ticker, date)",
    "CREATE INDEX IF NOT EXISTS idx_targets_daily_ticker_date ON targets_daily(ticker, date)",
]


@contextmanager
def connect_database(path: Path | str = DEFAULT_DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(path: Path | str = DEFAULT_DATABASE_PATH) -> None:
    with connect_database(path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)


def upsert_tickers(frame: pd.DataFrame, path: Path | str = DEFAULT_DATABASE_PATH) -> None:
    columns = ["ticker", "name", "exchange", "sector", "industry", "asset_type", "is_active"]
    payload = frame.copy()
    for column in columns:
        if column not in payload.columns:
            payload[column] = None
    payload["ticker"] = payload["ticker"].str.upper()
    payload["asset_type"] = payload["asset_type"].fillna("stock")
    payload["is_active"] = payload["is_active"].fillna(1).astype(int)

    sql = """
        INSERT INTO tickers (
            ticker, name, exchange, sector, industry, asset_type, is_active
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            name = excluded.name,
            exchange = excluded.exchange,
            sector = excluded.sector,
            industry = excluded.industry,
            asset_type = excluded.asset_type,
            is_active = excluded.is_active,
            updated_at = CURRENT_TIMESTAMP
    """
    records = payload[columns].itertuples(index=False, name=None)
    with connect_database(path) as connection:
        connection.executemany(sql, records)


def upsert_prices_daily(
    frame: pd.DataFrame,
    path: Path | str = DEFAULT_DATABASE_PATH,
    source: str = "manual",
) -> None:
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    payload = frame.copy()
    payload["date"] = pd.to_datetime(payload["date"]).dt.strftime("%Y-%m-%d")
    payload["ticker"] = payload["ticker"].str.upper()
    if "adjusted_close" not in payload.columns:
        payload["adjusted_close"] = payload["close"]
    payload["source"] = source

    columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
    ]
    sql = """
        INSERT INTO prices_daily (
            date, ticker, open, high, low, close, adjusted_close, volume, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, ticker) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            adjusted_close = excluded.adjusted_close,
            volume = excluded.volume,
            source = excluded.source
    """
    records = payload[columns].itertuples(index=False, name=None)
    with connect_database(path) as connection:
        connection.executemany(sql, records)


def read_prices_daily(
    path: Path | str = DEFAULT_DATABASE_PATH,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    query = "SELECT date, ticker, open, high, low, close, adjusted_close, volume FROM prices_daily"
    clauses = []
    params: list[str] = []

    if tickers:
        placeholders = ", ".join("?" for _ in tickers)
        clauses.append(f"ticker IN ({placeholders})")
        params.extend(ticker.upper() for ticker in tickers)
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY ticker, date"

    with connect_database(path) as connection:
        return pd.read_sql_query(query, connection, params=params, parse_dates=["date"])


def upsert_features_daily(
    frame: pd.DataFrame,
    path: Path | str = DEFAULT_DATABASE_PATH,
    horizon_days: int = 5,
) -> None:
    feature_columns, _, _ = modeling_columns(horizon_days)
    required = ["date", "ticker", *feature_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    payload = frame.dropna(subset=feature_columns).copy()
    payload["date"] = pd.to_datetime(payload["date"]).dt.strftime("%Y-%m-%d")
    payload["ticker"] = payload["ticker"].str.upper()

    columns = ["date", "ticker", *feature_columns]
    sql = """
        INSERT INTO features_daily (
            date, ticker, return_1d, return_5d, return_20d,
            volatility_5d, volatility_20d, close_to_ma_5, close_to_ma_20,
            volume_to_ma_20, intraday_range, close_position
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, ticker) DO UPDATE SET
            return_1d = excluded.return_1d,
            return_5d = excluded.return_5d,
            return_20d = excluded.return_20d,
            volatility_5d = excluded.volatility_5d,
            volatility_20d = excluded.volatility_20d,
            close_to_ma_5 = excluded.close_to_ma_5,
            close_to_ma_20 = excluded.close_to_ma_20,
            volume_to_ma_20 = excluded.volume_to_ma_20,
            intraday_range = excluded.intraday_range,
            close_position = excluded.close_position
    """
    records = payload[columns].itertuples(index=False, name=None)
    with connect_database(path) as connection:
        connection.executemany(sql, records)


def upsert_targets_daily(
    frame: pd.DataFrame,
    path: Path | str = DEFAULT_DATABASE_PATH,
    horizon_days: int = 5,
) -> None:
    _, target_return, target_close = modeling_columns(horizon_days)
    required = ["date", "ticker", target_close, target_return]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    payload = frame.dropna(subset=[target_close, target_return]).copy()
    payload["date"] = pd.to_datetime(payload["date"]).dt.strftime("%Y-%m-%d")
    payload["ticker"] = payload["ticker"].str.upper()
    payload["horizon_days"] = horizon_days
    payload["future_close"] = payload[target_close]
    payload["future_return"] = payload[target_return]

    columns = ["date", "ticker", "horizon_days", "future_close", "future_return"]
    sql = """
        INSERT INTO targets_daily (
            date, ticker, horizon_days, future_close, future_return
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date, ticker, horizon_days) DO UPDATE SET
            future_close = excluded.future_close,
            future_return = excluded.future_return
    """
    records = payload[columns].itertuples(index=False, name=None)
    with connect_database(path) as connection:
        connection.executemany(sql, records)


def read_modeling_frame(
    path: Path | str = DEFAULT_DATABASE_PATH,
    horizon_days: int = 5,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns, _, _ = modeling_columns(horizon_days)
    selected_features = ", ".join(f"f.{column}" for column in feature_columns)
    query = f"""
        SELECT
            f.date,
            f.ticker,
            p.close,
            {selected_features},
            t.future_close,
            t.future_return
        FROM features_daily f
        INNER JOIN targets_daily t
            ON f.date = t.date
            AND f.ticker = t.ticker
            AND t.horizon_days = ?
        INNER JOIN prices_daily p
            ON f.date = p.date
            AND f.ticker = p.ticker
    """
    params: list[object] = [horizon_days]
    if tickers:
        placeholders = ", ".join("?" for _ in tickers)
        query += f" WHERE f.ticker IN ({placeholders})"
        params.extend(ticker.upper() for ticker in tickers)
    query += " ORDER BY f.ticker, f.date"

    with connect_database(path) as connection:
        return pd.read_sql_query(query, connection, params=params, parse_dates=["date"])


def table_counts(path: Path | str = DEFAULT_DATABASE_PATH) -> dict[str, int]:
    tables = ["tickers", "prices_daily", "features_daily", "targets_daily"]
    with connect_database(path) as connection:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
