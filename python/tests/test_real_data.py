import pandas as pd
import pytest

from quantiq.data import load_real_ohlcv


class _FakeTicker:
    def __init__(self, history_df: pd.DataFrame):
        self._history_df = history_df

    def history(self, period: str, interval: str):  # noqa: ARG002 -- matches yfinance's signature
        return self._history_df


def _fake_yfinance_history(monkeypatch, df: pd.DataFrame):
    """Patches quantiq.data.yf.Ticker so tests never hit the real network --
    load_real_ohlcv's contract (column names/shape) is what's under test
    here, not Yahoo Finance's actual availability."""
    import quantiq.data as data_module

    monkeypatch.setattr(data_module.yf, "Ticker", lambda ticker: _FakeTicker(df))  # noqa: ARG005


def test_load_real_ohlcv_normalizes_columns_and_shape(monkeypatch):
    idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101, 102, 103, 104],
            "High": [101.0, 102, 103, 104, 105],
            "Low": [99.0, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
            "Dividends": [0.0] * 5,
            "Stock Splits": [0.0] * 5,
        },
        index=idx,
    )
    _fake_yfinance_history(monkeypatch, raw)

    df = load_real_ohlcv("AAPL", period="1y")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert df.index.tz is None  # tz-naive, matching generate_synthetic_ohlcv's contract
    assert df["close"].iloc[0] == 100.5


def test_load_real_ohlcv_raises_on_empty_result(monkeypatch):
    _fake_yfinance_history(monkeypatch, pd.DataFrame())
    with pytest.raises(ValueError, match="no data"):
        load_real_ohlcv("NOT-A-REAL-TICKER")


def test_load_real_ohlcv_is_a_drop_in_for_the_backtester(monkeypatch):
    """The whole point of matching generate_synthetic_ohlcv's contract is
    that run_backtest doesn't need to know which data source it's looking
    at -- prove that by actually running one."""
    from quantiq.engine import run_backtest
    from quantiq.strategies import MovingAverageCrossover

    idx = pd.date_range("2024-01-02", periods=150, freq="B", tz="America/New_York")
    close = pd.Series(range(150), dtype=float) + 100
    raw = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [1_000_000] * 150,
        },
        index=idx,
    )
    _fake_yfinance_history(monkeypatch, raw)

    prices = load_real_ohlcv("AAPL", period="1y")
    result = run_backtest(prices, MovingAverageCrossover())
    assert len(result.equity_curve) == 150
