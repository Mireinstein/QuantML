import pandas as pd
import pytest

from quantml.risk import RiskLimits, apply_risk_limits


def make_series(values, n=10):
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(values, index=idx)


def test_position_is_clipped_to_max_position():
    raw = make_series([0.9, -0.9, 0.5, -0.3, 0.2, 0.1, 0.0, -0.1, 0.2, 0.3])
    returns = make_series([0.0] * 10)
    limits = RiskLimits(max_position=0.5, max_drawdown=1.0)  # drawdown limit disabled for this test
    result = apply_risk_limits(raw, returns, limits, cost_bps=0.0)
    assert (result.positions.abs() <= 0.5 + 1e-9).all()
    assert result.positions.iloc[0] == pytest.approx(0.5)   # 0.9 clipped down
    assert result.positions.iloc[1] == pytest.approx(-0.5)  # -0.9 clipped down


def test_max_drawdown_kill_switch_halts_trading():
    # One big loss day should trip a 10% max-drawdown limit; everything
    # after must be forced flat even though the strategy wants to trade.
    raw = make_series([1.0] * 10)
    returns = make_series([-0.15] + [0.05] * 9)
    limits = RiskLimits(max_position=1.0, max_drawdown=0.10)
    result = apply_risk_limits(raw, returns, limits, cost_bps=0.0)

    assert result.breach_type == "max_drawdown"
    assert result.breach_date == raw.index[0]
    # Halted flag is True starting the bar AFTER the breach (the breach bar
    # itself still traded -- that's what caused the breach).
    assert not result.halted.iloc[0]
    assert result.halted.iloc[1:].all()
    # Every position after the breach bar must be forced flat.
    assert (result.positions.iloc[1:] == 0.0).all()


def test_max_daily_loss_kill_switch():
    raw = make_series([1.0] * 5, n=5)
    returns = make_series([-0.06, 0.01, 0.01, 0.01, 0.01], n=5)
    limits = RiskLimits(max_position=1.0, max_drawdown=1.0, max_daily_loss=0.05)
    result = apply_risk_limits(raw, returns, limits, cost_bps=0.0)

    assert result.breach_type == "max_daily_loss"
    assert result.breach_date == raw.index[0]
    assert (result.positions.iloc[1:] == 0.0).all()


def test_no_breach_when_within_limits():
    raw = make_series([0.2, -0.2, 0.1, -0.1, 0.0], n=5)
    returns = make_series([0.001, -0.001, 0.002, -0.002, 0.001], n=5)
    limits = RiskLimits(max_position=1.0, max_drawdown=0.5)
    result = apply_risk_limits(raw, returns, limits, cost_bps=0.0)

    assert result.breach_type is None
    assert result.breach_date is None
    assert not result.halted.any()


def test_equity_curve_matches_manual_calculation_when_no_costs():
    raw = make_series([1.0, 1.0, 1.0], n=3)
    returns = make_series([0.10, -0.05, 0.02], n=3)
    limits = RiskLimits(max_position=1.0, max_drawdown=1.0)
    result = apply_risk_limits(raw, returns, limits, cost_bps=0.0)

    expected = (1.10) * (0.95) * (1.02)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected)
