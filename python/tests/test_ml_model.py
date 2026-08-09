import numpy as np
import pandas as pd
import pytest

from quantiq.data import generate_synthetic_ohlcv
from quantiq.ml.features import FEATURE_COLUMNS, LABEL_COLUMN, build_features_and_labels
from quantiq.ml.model import (
    SklearnSignalModel,
    TorchSignalModel,
    build_gradient_boosting,
    build_logistic_baseline,
)


@pytest.fixture(scope="module")
def train_test_split():
    prices = generate_synthetic_ohlcv(n_days=400, seed=3)
    df = build_features_and_labels(prices)
    split = int(len(df) * 0.75)
    return df.iloc[:split], df.iloc[split:]


def test_sklearn_model_predict_proba_in_unit_interval(train_test_split):
    train_df, test_df = train_test_split
    model = SklearnSignalModel(build_logistic_baseline()).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    proba = model.predict_proba(test_df)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    assert len(proba) == len(test_df)


def test_sklearn_model_predict_before_fit_raises():
    model = SklearnSignalModel(build_gradient_boosting())
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame(columns=FEATURE_COLUMNS))


def test_sklearn_model_save_and_load_round_trip(train_test_split, tmp_path):
    train_df, test_df = train_test_split
    model = SklearnSignalModel(build_logistic_baseline()).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    path = tmp_path / "model.joblib"
    model.save(path)

    loaded = SklearnSignalModel.load(path)
    np.testing.assert_array_equal(model.predict_proba(test_df), loaded.predict_proba(test_df))


def test_torch_model_predict_proba_in_unit_interval(train_test_split):
    train_df, test_df = train_test_split
    model = TorchSignalModel(epochs=3).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    proba = model.predict_proba(test_df)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    assert len(proba) == len(test_df)


def test_torch_model_handles_fewer_rows_than_window_without_crashing(train_test_split):
    """Regression test: calling predict_proba with fewer rows than the
    model's sequence window used to crash (np.stack on an empty list) --
    this is exactly what a live single-ticker "predict today" endpoint can
    hit if it's ever passed less history than the window needs. Must
    degrade to neutral, not raise."""
    train_df, test_df = train_test_split
    model = TorchSignalModel(window=20, epochs=3).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    short = test_df[FEATURE_COLUMNS].iloc[:5]  # fewer than window=20
    proba = model.predict_proba(short)
    assert len(proba) == 5
    assert (proba == 0.5).all()


def test_torch_model_rows_before_first_window_are_neutral(train_test_split):
    train_df, _ = train_test_split
    model = TorchSignalModel(window=20, epochs=3).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    proba = model.predict_proba(train_df)
    # First `window - 1` rows can't form a full sequence yet -> neutral 0.5.
    assert (proba[:19] == 0.5).all()


def test_torch_model_save_and_load_round_trip(train_test_split, tmp_path):
    train_df, test_df = train_test_split
    model = TorchSignalModel(epochs=3).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    path = tmp_path / "model.pt"
    model.save(path)

    loaded = TorchSignalModel.load(path)
    np.testing.assert_allclose(model.predict_proba(test_df), loaded.predict_proba(test_df), rtol=1e-5)


def test_torch_model_predict_before_fit_raises():
    model = TorchSignalModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame(columns=FEATURE_COLUMNS))
