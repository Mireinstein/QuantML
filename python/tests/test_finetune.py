import pytest

from quantiq.finetune.data import LABEL_NAMES, load_sentiment_data
from quantiq.finetune.eval_harness import (
    MAX_ACCURACY_REGRESSION,
    MAX_F1_REGRESSION,
    MIN_ACCURACY,
    MIN_F1_MACRO,
    EvalResult,
    check,
)
from quantiq.finetune.model import FineTunedSentimentScorer, ModelNotTrainedError


def test_load_sentiment_data_respects_subsample_cap():
    data = load_sentiment_data(train_subsample=50)
    assert len(data.train) == 50
    assert len(data.validation) > 0  # validation set is never subsampled


def test_load_sentiment_data_labels_match_known_encoding():
    data = load_sentiment_data(train_subsample=20)
    assert set(data.train["label"]).issubset({0, 1, 2})
    assert LABEL_NAMES == ["Bearish", "Bullish", "Neutral"]


# --- Model: loaded once per module, since loading DistilBERT + the LoRA
# adapter takes a few seconds and every test below is read-only against it.


@pytest.fixture(scope="module")
def scorer():
    try:
        return FineTunedSentimentScorer()
    except ModelNotTrainedError:
        pytest.skip("No fine-tuned adapter present -- run `python -m quantiq.finetune.train` first.")


def test_predict_returns_a_valid_label_and_probability_simplex(scorer):
    result = scorer.predict("Company reports record profits, stock surges on strong earnings beat")
    assert result["label"] in LABEL_NAMES
    assert 0.0 <= result["score"] <= 1.0
    probs = result["probabilities"]
    assert set(probs) == set(LABEL_NAMES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_signal_value_bounded_in_unit_range(scorer):
    value = scorer.signal_value("Markets rally as tech earnings crush expectations")
    assert -1.0 <= value <= 1.0


# --- Eval harness gate logic (pure, no model loading needed) ---------------


def test_check_passes_a_good_result_with_no_baseline():
    result = EvalResult(accuracy=0.7, f1_macro=0.5, n_eval=500)
    assert check(result, baseline=None) == []


def test_check_fails_below_accuracy_floor():
    result = EvalResult(accuracy=MIN_ACCURACY - 0.05, f1_macro=0.5, n_eval=500)
    assert any("Accuracy" in f and "minimum" in f for f in check(result, baseline=None))


def test_check_fails_below_f1_floor():
    result = EvalResult(accuracy=0.7, f1_macro=MIN_F1_MACRO - 0.05, n_eval=500)
    assert any("F1" in f and "minimum" in f for f in check(result, baseline=None))


def test_check_fails_on_regression_versus_baseline():
    baseline = {"accuracy": 0.7, "f1_macro": 0.5}
    result = EvalResult(accuracy=0.7 - MAX_ACCURACY_REGRESSION - 0.01, f1_macro=0.5, n_eval=500)
    assert any("Accuracy regressed" in f for f in check(result, baseline))


def test_check_tolerates_small_drop_within_budget():
    baseline = {"accuracy": 0.7, "f1_macro": 0.5}
    result = EvalResult(
        accuracy=0.7 - MAX_ACCURACY_REGRESSION + 0.005,
        f1_macro=0.5 - MAX_F1_REGRESSION + 0.005,
        n_eval=500,
    )
    assert check(result, baseline) == []
