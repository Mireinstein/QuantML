import numpy as np
import pytest

from quantiq.execution import simulate_execution, summarize_execution
from quantiq.tick_data import generate_synthetic_ticks


def test_generate_synthetic_ticks_shape():
    ticks = generate_synthetic_ticks(n_ticks=1000, seed=1)
    assert len(ticks) == 1000
    assert ticks.dtype == np.int64


def test_flat_target_produces_no_orders():
    ticks = generate_synthetic_ticks(n_ticks=500, seed=1)
    targets = np.zeros(5)
    result = simulate_execution(ticks, targets, ticks_per_bar=100)
    assert result.target_delta == []


def test_nonzero_target_fills_against_book():
    ticks = generate_synthetic_ticks(n_ticks=500, seed=1)
    targets = np.array([1.0])  # go fully long
    result = simulate_execution(ticks, targets, ticks_per_bar=100, order_notional_qty=50)
    assert len(result.target_delta) == 1
    assert result.filled_delta[0] == result.target_delta[0]  # ample background liquidity -> full fill
    assert result.avg_fill_price[0] is not None


def test_buy_fills_at_or_above_mid_sell_at_or_below():
    ticks = generate_synthetic_ticks(n_ticks=1000, seed=3)
    # Two bars: go long, then flat (sell back to zero).
    targets = np.array([1.0, 0.0])
    result = simulate_execution(ticks, targets, ticks_per_bar=200, order_notional_qty=50)
    assert len(result.target_delta) == 2
    buy_price = result.avg_fill_price[0]
    buy_mid = result.mid_price_at_submit[0]
    assert buy_price >= buy_mid  # crossing the book to buy costs at least the ask

    sell_price = result.avg_fill_price[1]
    sell_mid = result.mid_price_at_submit[1]
    assert sell_price <= sell_mid  # crossing the book to sell nets at most the bid


def test_slippage_is_nonnegative_when_crossing_spread():
    ticks = generate_synthetic_ticks(n_ticks=500, seed=5)
    targets = np.array([1.0])
    result = simulate_execution(ticks, targets, ticks_per_bar=100, order_notional_qty=50, spread_ticks=3)
    assert result.slippage_bps[0] > 0  # paid through the spread to get filled


def test_summarize_execution_keys():
    ticks = generate_synthetic_ticks(n_ticks=500, seed=1)
    targets = np.array([1.0, -1.0, 0.5])
    result = simulate_execution(ticks, targets, ticks_per_bar=150, order_notional_qty=30)
    summary = summarize_execution(result)
    assert set(summary.keys()) == {"orders_submitted", "fill_rate", "avg_slippage_bps", "max_slippage_bps"}
    assert summary["fill_rate"] == 1.0
