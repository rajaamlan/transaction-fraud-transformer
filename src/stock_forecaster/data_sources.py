from __future__ import annotations

from datetime import date

import pandas as pd


YFINANCE_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
}


def normalize_yfinance_prices(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize a yfinance download into the project's daily price schema."""

    if downloaded.empty:
        return pd.DataFrame(columns=["date", "ticker", *YFINANCE_COLUMN_MAP.values()])

    tickers = [ticker.upper() for ticker in tickers]
    frames = []

    if isinstance(downloaded.columns, pd.MultiIndex):
        data = downloaded.copy()
        level_0_values = set(str(value) for value in data.columns.get_level_values(0))
        if set(YFINANCE_COLUMN_MAP).intersection(level_0_values):
            data = data.swaplevel(axis=1).sort_index(axis=1)

        available_tickers = set(str(value).upper() for value in data.columns.get_level_values(0))
        for ticker in tickers:
            if ticker not in available_tickers:
                continue
            ticker_frame = data[ticker].copy()
            frames.append(_normalize_single_ticker_frame(ticker_frame, ticker))
    else:
        if len(tickers) != 1:
            raise ValueError("Flat yfinance data can only be normalized for one ticker.")
        frames.append(_normalize_single_ticker_frame(downloaded.copy(), tickers[0]))

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", *YFINANCE_COLUMN_MAP.values()])

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.dropna(subset=["open", "high", "low", "close", "volume"])
    prices["volume"] = prices["volume"].astype("int64")
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_yfinance_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV prices from yfinance.

    yfinance is a convenient starter source, but it is not an institutional data
    vendor. Keep the source column so better feeds can be compared later.
    """

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to download starter market data.") from exc

    end_date = end_date or date.today().isoformat()
    downloaded = yf.download(
        tickers=[ticker.upper() for ticker in tickers],
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="ticker",
        actions=False,
        progress=False,
        threads=True,
    )
    return normalize_yfinance_prices(downloaded, tickers)


def _normalize_single_ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = frame.rename(columns=YFINANCE_COLUMN_MAP)
    expected = ["open", "high", "low", "close", "adjusted_close", "volume"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Downloaded data for {ticker} is missing columns: {missing}")

    normalized = frame[expected].copy()
    normalized.insert(0, "ticker", ticker.upper())
    normalized.insert(0, "date", pd.to_datetime(normalized.index).date)
    return normalized

