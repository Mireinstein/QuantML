"""Trading strategies. Each strategy maps price (and optionally an external
signal) history to a target position in [-1, 1] (short .. flat .. long) for
the next bar."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
