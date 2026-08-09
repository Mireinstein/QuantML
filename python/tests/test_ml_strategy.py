import numpy as np
import pandas as pd
import pytest

from quantml.data import generate_synthetic_ohlcv
from quantml.strategies import MLSignalStrategy


class _StubModel:
    """Always predicts the same probability, so we can check the
    proba->position mapping and the shift(1) no-lookahead behavior in
    isolation from any real model's training noise."""

    def __init__(self, proba: float):
        self.proba = proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.proba)


@pytest.fixture
def prices():
    return generate_synthetic_ohlcv(n_days=200, seed=2)


def test_confident_up_prediction_maps_to_near_full_long(prices):
    strategy = MLSignalStrategy(model=_StubModel(0.95))
    pos = strategy.positions(prices)
    # 2*0.95 - 1 = 0.9, shifted by one day.
    nonzero = pos[pos != 0]
    assert (nonzero.round(2) == 0.9).all()


def test_confident_down_prediction_maps_to_near_full_short(prices):
    strategy = MLSignalStrategy(model=_StubModel(0.05))
    pos = strategy.positions(prices)
    nonzero = pos[pos != 0]
    assert (nonzero.round(2) == -0.9).all()


def test_neutral_prediction_maps_to_flat(prices):
    strategy = MLSignalStrategy(model=_StubModel(0.5))
    pos = strategy.positions(prices)
    assert (pos == 0.0).all()


def test_position_is_shifted_one_day_no_lookahead(prices):
    """A varying (not constant) model output must show up in the position
    series one day LATER than the day the features were computed for --
    that's what makes it safe to hold, since it was decided using only
    information known before that day opened."""

    class _VaryingModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            # A distinctive, index-dependent value so we can trace exactly
            # which day's prediction ends up on which day's position.
            return np.linspace(0.1, 0.9, len(X))

    from quantml.ml.features import build_features

    strategy = MLSignalStrategy(model=_VaryingModel())
    pos = strategy.positions(prices)

    features = build_features(prices)
    raw_signal = pd.Series(2 * _VaryingModel().predict_proba(features) - 1, index=features.index)

    common_dates = features.index[1:]  # skip the first: nothing to shift in from before it
    for date in common_dates[:10]:
        prior_date = features.index[features.index.get_loc(date) - 1]
        assert pos.loc[date] == pytest.approx(raw_signal.loc[prior_date])


def test_positions_are_bounded(prices):
    strategy = MLSignalStrategy(model=_StubModel(1.0))
    pos = strategy.positions(prices)
    assert pos.between(-1.0, 1.0).all()
