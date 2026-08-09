"""OHLCV price data for backtesting: a synthetic generator (no external
dependency, always available) and a real-market-data loader (Yahoo Finance
via yfinance, no API key required). Both return the same column contract
(open/high/low/close/volume, tz-naive DatetimeIndex), so every consumer --
the backtester, walk-forward evaluation, VaR, GARCH, execution sim, and the
web dashboard -- works identically against either without caring which one
it's looking at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def generate_synthetic_ohlcv(
    n_days: int = 750,
    start_price: float = 100.0,
    annual_drift: float = 0.06,
    annual_vol: float = 0.25,
    seed: int = 42,
    start_date: str = "2023-01-02",
) -> pd.DataFrame:
    """Generates a synthetic daily OHLCV series via geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    drift = (annual_drift - 0.5 * annual_vol**2) * dt
    shocks = rng.normal(loc=drift, scale=annual_vol * np.sqrt(dt), size=n_days)
    close = start_price * np.exp(np.cumsum(shocks))

    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1]

    intraday_vol = annual_vol * np.sqrt(dt) * 0.5
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, intraday_vol, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, intraday_vol, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days)

    dates = pd.bdate_range(start=start_date, periods=n_days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def load_real_ohlcv(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Fetches REAL historical daily OHLCV for `ticker` from Yahoo Finance
    via yfinance -- free, no account or API key needed. Returns the same
    open/high/low/close/volume columns (lowercased) and a tz-naive
    DatetimeIndex as generate_synthetic_ohlcv, so it's a drop-in
    replacement anywhere that function is used.

    `period`/`interval` follow yfinance's own conventions (e.g. period:
    "1y", "3y", "max"; interval: "1d", "1wk", "1mo").
    """
    raw = yf.Ticker(ticker).history(period=period, interval=interval)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker {ticker!r} (period={period!r})")

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = None
    return df
