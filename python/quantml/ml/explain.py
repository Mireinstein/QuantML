"""Model explainability: which features the CURRENTLY LIVE model actually
relies on, measured directly rather than assumed.

Uses permutation importance (`sklearn.inspection.permutation_importance`):
for each feature, shuffle its values across the eval set and measure how
much the model's ROC-AUC drops -- a feature the model genuinely depends on
causes a real drop when scrambled; an unused or redundant one doesn't.
Model-agnostic (works identically for the sklearn wrappers and the
PyTorch GRU) since it only calls `predict_proba`, never looks inside the
model -- no SHAP/LIME dependency needed for a signal this reliable on a
model this size (8 features, not hundreds).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .features import FEATURE_COLUMNS


@dataclass
class FeatureImportance:
    feature: str
    importance_mean: float
    importance_std: float


def _auc_scorer(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    # permutation_importance calls this with `estimator` as whatever object
    # was passed in -- our model wrappers expose predict_proba, not a
    # scikit-learn-compatible .score(), so a custom scorer is required.
    proba = estimator.predict_proba(X)
    if len(np.unique(y)) < 2:
        # A shuffled/degenerate window can end up single-class; AUC is
        # undefined there, not a real zero -- report as no-signal rather
        # than crashing the whole importance computation.
        return 0.5
    return float(roc_auc_score(y, proba))


def explain_model(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, seed: int = 0) -> list[FeatureImportance]:
    """Ranks FEATURE_COLUMNS by how much shuffling each one degrades the
    model's held-out AUC. `X`/`y` should be data the model was NOT
    selected/tuned against (fresh eval data, same spirit as
    ml/eval_harness.py) so the ranking reflects genuine reliance, not an
    artifact of overfitting to this specific window."""
    result = permutation_importance(
        model, X[FEATURE_COLUMNS], y, scoring=_auc_scorer, n_repeats=n_repeats, random_state=seed
    )
    ranked = sorted(
        (
            FeatureImportance(feature=f, importance_mean=float(m), importance_std=float(s))
            for f, m, s in zip(FEATURE_COLUMNS, result.importances_mean, result.importances_std)
        ),
        key=lambda fi: fi.importance_mean,
        reverse=True,
    )
    return ranked
