import pytest

from quantiq.ml.eval_harness import (
    MAX_AUC_REGRESSION,
    MAX_SHARPE_REGRESSION,
    MIN_AUC,
    MIN_SHARPE,
    EvalResult,
    check,
)


def test_check_passes_a_good_result_with_no_baseline():
    result = EvalResult(auc=0.6, sharpe=1.0, n_eval=500)
    assert check(result, baseline=None) == []


def test_check_fails_below_auc_floor():
    result = EvalResult(auc=MIN_AUC - 0.01, sharpe=1.0, n_eval=500)
    assert any("AUC" in f and "minimum" in f for f in check(result, baseline=None))


def test_check_fails_below_sharpe_floor():
    result = EvalResult(auc=0.6, sharpe=MIN_SHARPE - 0.1, n_eval=500)
    assert any("Sharpe" in f and "minimum" in f for f in check(result, baseline=None))


def test_check_fails_on_auc_regression():
    baseline = {"auc": 0.6, "sharpe": 1.0}
    result = EvalResult(auc=0.6 - MAX_AUC_REGRESSION - 0.01, sharpe=1.0, n_eval=500)
    assert any("AUC regressed" in f for f in check(result, baseline))


def test_check_fails_on_sharpe_regression():
    baseline = {"auc": 0.6, "sharpe": 1.0}
    result = EvalResult(auc=0.6, sharpe=1.0 - MAX_SHARPE_REGRESSION - 0.01, n_eval=500)
    assert any("Sharpe regressed" in f for f in check(result, baseline))


def test_check_tolerates_small_drops_within_budget():
    baseline = {"auc": 0.6, "sharpe": 1.0}
    result = EvalResult(auc=0.6 - MAX_AUC_REGRESSION + 0.005, sharpe=1.0 - MAX_SHARPE_REGRESSION + 0.05, n_eval=500)
    assert check(result, baseline) == []
