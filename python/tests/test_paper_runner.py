from unittest.mock import patch

import pandas as pd
import pytest

from quantiq import paper_runner
from quantiq.paper_trading import OrderResult, Position
from quantiq.strategies import Strategy


class _FixedStrategy(Strategy):
    """A strategy stub whose target position is fixed, so rebalance()
    tests aren't coupled to real signal-generation logic -- only to
    paper_runner's own rebalance arithmetic and order-routing."""

    name = "fixed"

    def __init__(self, target: float):
        self.target = target

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series([self.target] * len(prices), index=prices.index)


@pytest.fixture
def fake_prices():
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}, index=idx)


def test_compute_target_shares_scales_position_by_qty_per_unit(fake_prices):
    with patch.object(paper_runner, "load_real_ohlcv", return_value=fake_prices):
        shares, last_close = paper_runner.compute_target_shares("AAPL", _FixedStrategy(0.5), qty_per_unit=20)
    assert shares == 10  # 0.5 * 20
    assert last_close == 100.0


def test_rebalance_submits_a_buy_when_below_target(fake_prices):
    with patch.object(paper_runner, "load_real_ohlcv", return_value=fake_prices), \
         patch.object(paper_runner, "get_position", return_value=None), \
         patch.object(
             paper_runner,
             "submit_market_order",
             return_value=OrderResult(id="1", symbol="AAPL", qty=10, side="buy", status="accepted"),
         ) as mock_submit:
        result = paper_runner.rebalance("AAPL", _FixedStrategy(0.5), qty_per_unit=20)

    assert result["current_shares"] == 0
    assert result["target_shares"] == 10
    assert result["delta"] == 10
    mock_submit.assert_called_once_with("AAPL", qty=10, side="buy")


def test_rebalance_submits_a_sell_when_above_target(fake_prices):
    existing = Position(symbol="AAPL", qty=15, avg_entry_price=100.0, market_value=1500.0, unrealized_pl=0.0)
    with patch.object(paper_runner, "load_real_ohlcv", return_value=fake_prices), \
         patch.object(paper_runner, "get_position", return_value=existing), \
         patch.object(
             paper_runner,
             "submit_market_order",
             return_value=OrderResult(id="2", symbol="AAPL", qty=5, side="sell", status="accepted"),
         ) as mock_submit:
        result = paper_runner.rebalance("AAPL", _FixedStrategy(0.5), qty_per_unit=20)

    assert result["delta"] == -5  # target 10, currently holding 15
    mock_submit.assert_called_once_with("AAPL", qty=5, side="sell")


def test_rebalance_submits_nothing_when_already_at_target(fake_prices):
    existing = Position(symbol="AAPL", qty=10, avg_entry_price=100.0, market_value=1000.0, unrealized_pl=0.0)
    with patch.object(paper_runner, "load_real_ohlcv", return_value=fake_prices), \
         patch.object(paper_runner, "get_position", return_value=existing), \
         patch.object(paper_runner, "submit_market_order") as mock_submit:
        result = paper_runner.rebalance("AAPL", _FixedStrategy(0.5), qty_per_unit=20)

    assert result["delta"] == 0
    assert result["order"] is None
    mock_submit.assert_not_called()


def test_dry_run_never_calls_submit_market_order(fake_prices):
    with patch.object(paper_runner, "load_real_ohlcv", return_value=fake_prices), \
         patch.object(paper_runner, "get_position", return_value=None), \
         patch.object(paper_runner, "submit_market_order") as mock_submit:
        result = paper_runner.rebalance("AAPL", _FixedStrategy(0.5), qty_per_unit=20, dry_run=True)

    mock_submit.assert_not_called()
    assert "DRY RUN" in result["order"]
