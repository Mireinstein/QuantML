"""Risk management: position limits and a max-drawdown / max-daily-loss
kill switch applied on top of a strategy's raw target positions. This is
enforcement, not just measurement -- Monte Carlo VaR/CVaR in metrics.py
measures risk; this module actually forces the strategy flat once a limit
is breached, and keeps it flat (it does not resume trading on its own).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskLimits:
    max_position: float = 1.0
    max_drawdown: float = 0.20
    max_daily_loss: float | None = None


@dataclass
class RiskManagedResult:
    positions: pd.Series   # positions AFTER limits/kill switch are applied
    returns: pd.Series
    equity_curve: pd.Series
    halted: pd.Series      # True on bars where the kill switch had already tripped
    breach_date: object
    breach_type: str | None  # "max_drawdown" | "max_daily_loss" | None


def apply_risk_limits(
    raw_positions: pd.Series,
    asset_returns: pd.Series,
    limits: RiskLimits,
    cost_bps: float = 5.0,
) -> RiskManagedResult:
    """Walks forward bar-by-bar rather than vectorizing, because the kill
    switch is inherently path-dependent: once it trips, every subsequent
    position is forced flat regardless of what the strategy wants next,
    which has to be computed sequentially to be correct.
    """
    idx = raw_positions.index
    clipped = raw_positions.clip(-limits.max_position, limits.max_position)

    positions = pd.Series(0.0, index=idx)
    returns = pd.Series(0.0, index=idx)
    equity_curve = pd.Series(1.0, index=idx)
    halted = pd.Series(False, index=idx)

    equity = 1.0
    peak_equity = 1.0
    is_halted = False
    breach_date = None
    breach_type = None
    prev_pos = 0.0

    for date in idx:
        pos = 0.0 if is_halted else float(clipped.loc[date])
        turnover = abs(pos - prev_pos)
        cost = turnover * (cost_bps / 10_000.0)
        ret = pos * float(asset_returns.loc[date]) - cost
        equity *= 1 + ret
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0

        positions.loc[date] = pos
        returns.loc[date] = ret
        equity_curve.loc[date] = equity
        halted.loc[date] = is_halted
        prev_pos = pos

        if not is_halted:
            if limits.max_daily_loss is not None and ret <= -limits.max_daily_loss:
                is_halted = True
                breach_date, breach_type = date, "max_daily_loss"
            elif drawdown <= -limits.max_drawdown:
                is_halted = True
                breach_date, breach_type = date, "max_drawdown"

    return RiskManagedResult(positions, returns, equity_curve, halted, breach_date, breach_type)
