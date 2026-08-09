"""Performance metrics for a backtest's return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / std)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    n_years = len(equity_curve) / TRADING_DAYS
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0:
        return -1.0
    return float(total_return ** (1 / n_years) - 1)


def win_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def monte_carlo_var(
    returns: pd.Series,
    horizon_days: int = 10,
    n_sims: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Historical-bootstrap Monte Carlo VaR/CVaR.

    Resamples (with replacement) `horizon_days` daily returns from the
    strategy's own historical return series `n_sims` times, compounds each
    simulated path, and reports the loss threshold not exceeded with
    `confidence` probability (VaR) and the average loss in the tail beyond
    it (CVaR/Expected Shortfall). Because it resamples the strategy's own
    realized daily returns rather than assuming a parametric distribution,
    it captures whatever skew/fat-tails are actually in the return series.
    """
    daily = returns.dropna().to_numpy()
    if len(daily) < 2:
        raise ValueError("Need at least 2 return observations to bootstrap from")

    rng = np.random.default_rng(seed)
    sampled = rng.choice(daily, size=(n_sims, horizon_days), replace=True)
    path_returns = np.prod(1 + sampled, axis=1) - 1.0

    alpha = 1 - confidence
    var = float(-np.quantile(path_returns, alpha))
    tail = path_returns[path_returns <= np.quantile(path_returns, alpha)]
    cvar = float(-tail.mean()) if len(tail) else var

    return {
        "horizon_days": horizon_days,
        "confidence": confidence,
        "var": round(var, 4),
        "cvar": round(cvar, 4),
        "n_sims": n_sims,
    }


def summarize(equity_curve: pd.Series, returns: pd.Series) -> dict:
    return {
        "sharpe": round(sharpe_ratio(returns), 3),
        "cagr": round(cagr(equity_curve), 4),
        "max_drawdown": round(max_drawdown(equity_curve), 4),
        "win_rate": round(win_rate(returns), 3),
        "final_equity": round(float(equity_curve.iloc[-1]), 4),
    }
