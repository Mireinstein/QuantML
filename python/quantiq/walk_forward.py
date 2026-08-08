"""Walk-forward backtest evaluation.

A single full-period Sharpe ratio can be dominated by one lucky or unlucky
stretch. Walk-forward evaluation instead splits the price history into
sequential out-of-sample folds and reports how performance holds up
fold-to-fold -- a strategy that only worked in one regime shows up here as
high variance across folds, not just a single misleadingly-good headline
number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import run_backtest
from .metrics import cagr, max_drawdown, sharpe_ratio
from .strategies import Strategy


@dataclass
class WalkForwardResult:
    fold_sharpe: list
    fold_cagr: list
    fold_max_drawdown: list
    fold_bounds: list  # (start_date, end_date) per fold


def run_walk_forward(
    prices: pd.DataFrame,
    strategy: Strategy,
    n_folds: int = 5,
    min_history: int = 100,
    cost_bps: float = 5.0,
) -> WalkForwardResult:
    """Runs `strategy` on the full price history (so rolling indicators have
    the lookback they need), then scores it fold-by-fold on sequential,
    non-overlapping out-of-sample windows after `min_history` warmup bars.
    """
    n = len(prices)
    if n <= min_history:
        raise ValueError(f"Not enough history ({n} bars) for min_history={min_history}")

    result = run_backtest(prices, strategy, cost_bps=cost_bps)

    fold_bounds = np.linspace(min_history, n, n_folds + 1, dtype=int)
    out = WalkForwardResult([], [], [], [])

    for i in range(n_folds):
        start, end = fold_bounds[i], fold_bounds[i + 1]
        if end <= start:
            continue
        fold_returns = result.returns.iloc[start:end]
        fold_equity = (1 + fold_returns).cumprod()

        out.fold_sharpe.append(sharpe_ratio(fold_returns))
        out.fold_cagr.append(cagr(fold_equity))
        out.fold_max_drawdown.append(max_drawdown(fold_equity))
        out.fold_bounds.append((prices.index[start], prices.index[end - 1]))

    return out


def summarize_walk_forward(result: WalkForwardResult) -> dict:
    sharpes = np.array(result.fold_sharpe)
    return {
        "n_folds": len(result.fold_sharpe),
        "mean_sharpe": round(float(sharpes.mean()), 3) if len(sharpes) else None,
        "std_sharpe": round(float(sharpes.std()), 3) if len(sharpes) else None,
        "worst_fold_sharpe": round(float(sharpes.min()), 3) if len(sharpes) else None,
        "worst_fold_drawdown": round(float(min(result.fold_max_drawdown)), 4)
        if result.fold_max_drawdown
        else None,
    }
