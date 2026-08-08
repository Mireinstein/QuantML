"""Synthetic tick-level mid-price data for order-book execution simulation.

Unlike data.py's daily OHLCV bars (used for the vectorized backtester),
this generates a high-frequency price path so strategy orders can be
submitted to the real C++ OrderBook and filled realistically -- with
slippage -- rather than assumed to fill at a daily closing price.
"""
from __future__ import annotations

import numpy as np


def generate_synthetic_ticks(
    n_ticks: int = 5000,
    start_price: int = 10_000,
    tick_vol: float = 2.0,
    seed: int = 7,
) -> np.ndarray:
    """Returns an array of n_ticks integer mid-prices via a random walk.

    Prices are integers (order-book "ticks"), matching the C++ OrderBook's
    integer price scale.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=tick_vol, size=n_ticks)
    path = start_price + np.cumsum(steps)
    return np.round(path).astype(np.int64)
