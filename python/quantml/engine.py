"""Vectorized backtest engine: turns a strategy's target positions into a
daily equity curve, net of transaction costs."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategies import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    turnover: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    cost_bps: float = 5.0,
) -> BacktestResult:
    positions = strategy.positions(prices)
    asset_returns = prices["close"].pct_change().fillna(0.0)

    turnover = positions.diff().abs().fillna(positions.abs())
    cost = turnover * (cost_bps / 10_000.0)

    strategy_returns = positions * asset_returns - cost
    equity_curve = (1 + strategy_returns).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        returns=strategy_returns,
        positions=positions,
        turnover=turnover,
    )
