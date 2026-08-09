import numpy as np
import pandas as pd
import pytest

from quantml.data import generate_synthetic_ohlcv
from quantml.engine import run_backtest
from quantml.metrics import cagr, max_drawdown, monte_carlo_var, sharpe_ratio, summarize
from quantml.strategies import MeanReversion, MovingAverageCrossover, SignalOverlayStrategy


@pytest.fixture
def prices():
    return generate_synthetic_ohlcv(n_days=300, seed=1)


def test_generate_synthetic_ohlcv_shape():
    df = generate_synthetic_ohlcv(n_days=100, seed=1)
    assert len(df) == 100
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df["high"] >= df["low"]).all()


def test_ma_crossover_positions_bounded(prices):
    pos = MovingAverageCrossover().positions(prices)
    assert pos.isin([-1.0, 0.0, 1.0]).all()


def test_ma_crossover_no_lookahead(prices):
    # Position on day t must not depend on day t's own close (shift(1) applied).
    strat = MovingAverageCrossover(fast=5, slow=20)
    pos = strat.positions(prices)
    truncated = prices.iloc[:-1]
    pos_truncated = strat.positions(truncated)
    pd.testing.assert_series_equal(pos.iloc[:-1], pos_truncated)


def test_mean_reversion_bounded(prices):
    pos = MeanReversion().positions(prices)
    assert (pos >= -1.0).all() and (pos <= 1.0).all()


def test_run_backtest_produces_equity_curve(prices):
    result = run_backtest(prices, MovingAverageCrossover())
    assert len(result.equity_curve) == len(prices)
    assert (result.equity_curve > 0).all()  # no blowups with these position sizes


def test_signal_overlay_bounded(prices):
    signal = pd.Series(np.random.default_rng(0).uniform(-1, 1, len(prices)), index=prices.index)
    strat = SignalOverlayStrategy(base=MovingAverageCrossover(), signal=signal, weight=0.5)
    pos = strat.positions(prices)
    assert (pos >= -1.0).all() and (pos <= 1.0).all()


def test_metrics_flat_returns_zero_sharpe():
    returns = pd.Series([0.0] * 50)
    assert sharpe_ratio(returns) == 0.0


def test_metrics_max_drawdown_is_negative_or_zero(prices):
    result = run_backtest(prices, MovingAverageCrossover())
    assert max_drawdown(result.equity_curve) <= 0.0


def test_summarize_keys(prices):
    result = run_backtest(prices, MovingAverageCrossover())
    summary = summarize(result.equity_curve, result.returns)
    assert set(summary.keys()) == {"sharpe", "cagr", "max_drawdown", "win_rate", "final_equity"}


def test_monte_carlo_var_keys_and_bounds(prices):
    result = run_backtest(prices, MovingAverageCrossover())
    mc = monte_carlo_var(result.returns, horizon_days=10, n_sims=2000, seed=1)
    assert set(mc.keys()) == {"horizon_days", "confidence", "var", "cvar", "n_sims"}
    assert mc["cvar"] >= mc["var"]  # expected shortfall is at least as bad as the VaR threshold


def test_monte_carlo_var_deterministic_given_seed(prices):
    result = run_backtest(prices, MovingAverageCrossover())
    mc1 = monte_carlo_var(result.returns, n_sims=1000, seed=7)
    mc2 = monte_carlo_var(result.returns, n_sims=1000, seed=7)
    assert mc1 == mc2


def test_monte_carlo_var_too_short_series_raises():
    with pytest.raises(ValueError):
        monte_carlo_var(pd.Series([0.01]))
