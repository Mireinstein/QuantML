"""Synthetic OHLCV price data for backtesting demos.

This is randomly generated (geometric Brownian motion), NOT real market
data. It exists so the backtester and RAG pipeline are fully runnable
end-to-end without any external data provider or API key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
