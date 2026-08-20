from quantml.ml.train import select_best


def test_select_best_prefers_sharpe_among_auc_eligible_candidates():
    """A below-chance classifier must not win selection on Sharpe alone --
    the eval gate downstream would reject it anyway (AUC floor 0.50), so
    selection restricts to candidates that can actually pass."""
    candidates = {
        "memorizer": (object(), 0.45, {"sharpe": 9.9}),
        "decent": (object(), 0.55, {"sharpe": 1.2}),
        "weak": (object(), 0.53, {"sharpe": 0.4}),
    }
    name, model, auc, bt = select_best(candidates)
    assert name == "decent"
    assert auc == 0.55


def test_select_best_falls_back_to_all_candidates_when_none_clear_floor():
    candidates = {
        "a": (object(), 0.45, {"sharpe": 1.0}),
        "b": (object(), 0.48, {"sharpe": 2.0}),
    }
    name, *_ = select_best(candidates)
    assert name == "b"
