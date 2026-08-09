"""Trading strategies. Each strategy maps price (and optionally an external
signal) history to a target position in [-1, 1] (short .. flat .. long) for
the next bar."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .ml.features import build_features


class Strategy:
    name: str = "base"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


@dataclass
class MovingAverageCrossover(Strategy):
    fast: int = 20
    slow: int = 100
    name: str = "ma_crossover"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        fast_ma = prices["close"].rolling(self.fast).mean()
        slow_ma = prices["close"].rolling(self.slow).mean()
        pos = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=prices.index)
        pos[fast_ma.isna() | slow_ma.isna()] = 0.0
        return pos.shift(1).fillna(0.0)  # trade on next bar, avoid lookahead


@dataclass
class MeanReversion(Strategy):
    window: int = 20
    z_entry: float = 1.0
    name: str = "mean_reversion"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        mean = prices["close"].rolling(self.window).mean()
        std = prices["close"].rolling(self.window).std()
        z = (prices["close"] - mean) / std
        pos = pd.Series(0.0, index=prices.index)
        pos[z > self.z_entry] = -1.0  # overbought -> short
        pos[z < -self.z_entry] = 1.0  # oversold -> long
        return pos.shift(1).fillna(0.0)


@dataclass
class SignalOverlayStrategy(Strategy):
    """Wraps a base strategy and tilts its position using an external signal
    series (e.g. a RAG-derived sentiment score in [-1, 1])."""

    base: Strategy
    signal: pd.Series
    weight: float = 0.5
    name: str = "signal_overlay"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        base_pos = self.base.positions(prices)
        sig = self.signal.reindex(prices.index).fillna(0.0).shift(1).fillna(0.0)
        combined = (1 - self.weight) * base_pos + self.weight * sig
        return combined.clip(-1.0, 1.0)


class _ProbaModel(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class MLSignalStrategy(Strategy):
    """A strategy driven by a trained model (see ml/model.py) instead of a
    hand-coded rule. The model predicts P(next day up) from ml/features.py's
    technical features; that probability is mapped to a position in
    [-1, 1] sized by the model's confidence (0.5 -> flat, 1.0 -> full long,
    0.0 -> full short) rather than a hard threshold, so a barely-confident
    prediction results in a small position, not the same full-size bet as
    a highly confident one.

    Same no-lookahead convention as every other strategy here: the model's
    prediction for day t (made from features computable using only data
    through day t) is shifted forward one day before being returned, so the
    position actually HELD on day t was decided using information known
    before day t opened.
    """

    model: Any = field(repr=False)  # duck-typed: needs predict_proba(features_df) -> np.ndarray
    name: str = "ml_signal"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        features = build_features(prices)
        proba_up = self.model.predict_proba(features)
        signal = pd.Series(2 * proba_up - 1, index=features.index).clip(-1.0, 1.0)
        return signal.reindex(prices.index).shift(1).fillna(0.0)
