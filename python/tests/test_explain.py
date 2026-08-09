import numpy as np
import pandas as pd

from quantml.data import generate_synthetic_ohlcv
from quantml.ml.explain import explain_model
from quantml.ml.features import FEATURE_COLUMNS, LABEL_COLUMN, build_features_and_labels
from quantml.ml.model import SklearnSignalModel, build_logistic_baseline


def test_explain_model_ranks_all_feature_columns():
    prices = generate_synthetic_ohlcv(n_days=400, seed=11)
    df = build_features_and_labels(prices)
    split = int(len(df) * 0.7)
    train_df, eval_df = df.iloc[:split], df.iloc[split:]

    model = SklearnSignalModel(build_logistic_baseline()).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    ranked = explain_model(model, eval_df, eval_df[LABEL_COLUMN], n_repeats=3, seed=0)

    assert {fi.feature for fi in ranked} == set(FEATURE_COLUMNS)
    assert len(ranked) == len(FEATURE_COLUMNS)
    # sorted descending by importance
    means = [fi.importance_mean for fi in ranked]
    assert means == sorted(means, reverse=True)


def test_explain_model_flags_an_informative_feature_above_pure_noise():
    """Build a dataset where the label is literally derived from one
    feature and nothing else -- that feature's permutation importance
    must come out highest, proving the ranking reflects real reliance."""
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    informative = rng.normal(size=n)
    noise_df = pd.DataFrame(
        {col: rng.normal(size=n) for col in FEATURE_COLUMNS if col != "return_1d"}, index=idx
    )
    noise_df["return_1d"] = informative
    label = (informative > 0).astype(int)
    df = noise_df.copy()
    df[LABEL_COLUMN] = label

    split = int(n * 0.6)
    train_df, eval_df = df.iloc[:split], df.iloc[split:]
    model = SklearnSignalModel(build_logistic_baseline()).fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])

    ranked = explain_model(model, eval_df, eval_df[LABEL_COLUMN], n_repeats=10, seed=0)
    assert ranked[0].feature == "return_1d"
