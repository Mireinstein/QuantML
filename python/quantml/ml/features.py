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
    "return_20d",
    "volatility_20d",
    "ma_ratio_10_50",
    "rsi_14",
    "volume_change_5d",
    "high_low_range",
    "macd_hist",
    "bollinger_pct_b",
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


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD line minus its signal line), expressed as a
    fraction of price so it's on the same relative scale as the other
    return-based features (and comparable across tickers at very
    different price levels -- relevant now that eval_harness.py can
    cross-check a model against a different ticker's data)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / close


def _bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """%B: where price sits within its rolling Bollinger Band, in [0, 1]
    under normal conditions (0 = at the lower band, 1 = at the upper band;
    can go slightly outside that range on a sharp move). A 0.5 fallback
    when the bands have zero width (flat price -- no real position to
    report) matches rsi_14's neutral-fallback convention above."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ((close - lower) / (upper - lower).replace(0, np.nan)).fillna(0.5)


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
    features["return_20d"] = close.pct_change(20)
    features["volatility_20d"] = returns.rolling(20).std()
    ma10 = close.rolling(10).mean()
    ma50 = close.rolling(50).mean()
    features["ma_ratio_10_50"] = ma10 / ma50 - 1
    features["rsi_14"] = _rsi(close, 14)
    volume = prices["volume"]
    features["volume_change_5d"] = volume / volume.rolling(5).mean() - 1
    features["high_low_range"] = (prices["high"] - prices["low"]) / close
    features["macd_hist"] = _macd_hist(close)
    features["bollinger_pct_b"] = _bollinger_pct_b(close)

    # inf can leak through ratio features (e.g. volume_change_5d when the
    # rolling mean volume is 0) and would survive dropna() untouched.
    return features.replace([np.inf, -np.inf], np.nan).dropna()


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
