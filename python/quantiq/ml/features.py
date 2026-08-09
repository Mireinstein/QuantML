"""Feature engineering for the ML trading signal: turns raw OHLCV into a
table of technical features plus a next-day-direction label, with the same
no-lookahead discipline as strategies.py -- every feature at row t is
computable from data available through the close of day t, and the label
at row t is about day t+1 (so a model trained on it, at inference time,
naturally predicts the next bar -- see strategies.py::MLSignalStrategy for
how that prediction becomes a position without leaking day t+1 into day t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_20d",
    "ma_ratio_10_50",
    "rsi_14",
    "volume_change_5d",
    "high_low_range",
]
LABEL_COLUMN = "next_day_up"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # neutral when there's no loss to divide by yet


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed like `prices`, with FEATURE_COLUMNS
    (early rows that don't have enough history for the longest rolling
    window are dropped, not filled with fake values)."""
    close = prices["close"]
    returns = close.pct_change()

    features = pd.DataFrame(index=prices.index)
    features["return_1d"] = returns
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["volatility_20d"] = returns.rolling(20).std()
    ma10 = close.rolling(10).mean()
    ma50 = close.rolling(50).mean()
    features["ma_ratio_10_50"] = ma10 / ma50 - 1
    features["rsi_14"] = _rsi(close, 14)
    volume = prices["volume"]
    features["volume_change_5d"] = volume / volume.rolling(5).mean() - 1
    features["high_low_range"] = (prices["high"] - prices["low"]) / close

    return features.dropna()


def build_features_and_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Features joined with the next-day-direction label, for training.
    The very last row is dropped -- there's no "next day" to label it with,
    and keeping it with a fake label would silently corrupt the training
    signal."""
    features = build_features(prices)
    next_return = prices["close"].pct_change().shift(-1)
    label = (next_return > 0).astype(int)

    df = features.copy()
    df[LABEL_COLUMN] = label.reindex(features.index)
    return df.iloc[:-1]  # drop the last row: its label needs a day that doesn't exist yet
