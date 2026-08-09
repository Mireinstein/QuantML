import numpy as np
import pandas as pd
import pytest

from quantiq.data import generate_synthetic_ohlcv
from quantiq.ml.features import FEATURE_COLUMNS, LABEL_COLUMN, build_features, build_features_and_labels


@pytest.fixture
def prices():
    return generate_synthetic_ohlcv(n_days=200, seed=1)


def test_build_features_has_expected_columns_and_no_nans(prices):
    features = build_features(prices)
    assert list(features.columns) == FEATURE_COLUMNS
    assert not features.isna().any().any()


def test_build_features_drops_only_the_warmup_period(prices):
    # The longest rolling window is 50 (ma_ratio_10_50); expect roughly
    # that many leading rows dropped, not the whole series.
    features = build_features(prices)
    assert len(features) > len(prices) - 60
    assert len(features) < len(prices)


def test_labels_are_derived_from_the_next_days_return(prices):
    df = build_features_and_labels(prices)
    close = prices["close"]
    for date in df.index[:20]:
        next_date = close.index[close.index.get_loc(date) + 1]
        expected_up = int(close.loc[next_date] > close.loc[date])
        assert df.loc[date, LABEL_COLUMN] == expected_up


def test_last_row_is_dropped_since_it_has_no_next_day(prices):
    features = build_features(prices)
    df = build_features_and_labels(prices)
    assert df.index[-1] == features.index[-2]


def test_rsi_is_bounded_zero_to_hundred(prices):
    features = build_features(prices)
    assert features["rsi_14"].between(0, 100).all()
