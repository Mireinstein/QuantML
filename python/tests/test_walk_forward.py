import pytest

from quantml.data import generate_synthetic_ohlcv
from quantml.strategies import MovingAverageCrossover
from quantml.walk_forward import run_walk_forward, summarize_walk_forward


@pytest.fixture
def prices():
    return generate_synthetic_ohlcv(n_days=500, seed=2)


def test_run_walk_forward_produces_n_folds(prices):
    result = run_walk_forward(prices, MovingAverageCrossover(), n_folds=5, min_history=100)
    assert len(result.fold_sharpe) == 5
    assert len(result.fold_cagr) == 5
    assert len(result.fold_max_drawdown) == 5
    assert len(result.fold_bounds) == 5


def test_fold_bounds_are_sequential_and_nonoverlapping(prices):
    result = run_walk_forward(prices, MovingAverageCrossover(), n_folds=4, min_history=100)
    ends = [end for _, end in result.fold_bounds]
    starts = [start for start, _ in result.fold_bounds]
    # Each fold's start should come after the previous fold's end.
    for i in range(1, len(starts)):
        assert starts[i] > ends[i - 1]


def test_insufficient_history_raises(prices):
    with pytest.raises(ValueError):
        run_walk_forward(prices, MovingAverageCrossover(), n_folds=5, min_history=10_000)


def test_summarize_walk_forward_keys(prices):
    result = run_walk_forward(prices, MovingAverageCrossover(), n_folds=5, min_history=100)
    summary = summarize_walk_forward(result)
    assert set(summary.keys()) == {
        "n_folds", "mean_sharpe", "std_sharpe", "worst_fold_sharpe", "worst_fold_drawdown",
    }
    assert summary["n_folds"] == 5
    assert summary["worst_fold_sharpe"] <= summary["mean_sharpe"]
