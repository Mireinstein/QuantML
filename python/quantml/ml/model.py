"""Models that predict next-day price direction from the features in
features.py. Two families, both exposed through the same
`predict_proba(features_df) -> np.ndarray` interface so strategies.py and
the training/eval scripts don't need to know which one they're holding:

- A classical scikit-learn baseline (logistic regression / gradient
  boosting) -- same pattern as TenantIQ's risk_model.py.
- A small PyTorch GRU that reads a rolling WINDOW of past feature rows as a
  sequence, rather than a single flat row -- a real (if modest) recurrent
  model over the temporal structure, not just sklearn-on-a-flat-vector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# sklearn MUST be imported before torch in this process: on this platform,
# importing torch first and then running scikit-learn's native
# (Cython/OpenMP) HistGradientBoostingClassifier code hard-crashes the
# interpreter (SIGSEGV) -- a known class of conflict between PyTorch's
# bundled OpenMP runtime and the one scikit-learn/numpy initialize, where
# whichever library's runtime registers first "wins" and the other
# corrupts state instead of coexisting. Importing sklearn's native pieces
# first avoids it entirely. Do not reorder these two import blocks.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn

from .features import FEATURE_COLUMNS

SEQUENCE_WINDOW = 20  # trading days of feature history the GRU reads per prediction


def build_logistic_baseline() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])


def build_gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, random_state=42)


class SklearnSignalModel:
    """Thin wrapper matching TenantIQ's RiskModel shape: fit/predict_proba/save/load."""

    def __init__(self, estimator=None):
        self.estimator = estimator or build_gradient_boosting()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnSignalModel":
        self.estimator.fit(X[FEATURE_COLUMNS], y)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SklearnSignalModel.fit() must be called before predict_proba()")
        return self.estimator.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump(self.estimator, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnSignalModel":
        model = cls(estimator=joblib.load(path))
        model._fitted = True
        return model


class GRUClassifier(nn.Module):
    """Reads a (window, n_features) sequence of daily features and outputs
    a single logit for P(next day up). Small on purpose: this is a portfolio
    demo trained on a few years of daily bars, not a production model --
    a large network would just overfit a dataset this size."""

    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)  # h_n: (1, batch, hidden_size) -- final hidden state
        return self.head(h_n.squeeze(0)).squeeze(-1)  # (batch,) raw logits


def _make_sequences(features: pd.DataFrame, window: int = SEQUENCE_WINDOW) -> tuple[np.ndarray, pd.Index]:
    """Turns a flat feature table into overlapping (window, n_features)
    sequences, one per row from `window` onward. Returned index aligns each
    sequence with the LAST day in its window (the day the prediction is
    "as of"), matching build_features_and_labels' label convention."""
    values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    sequences = np.stack([values[i - window : i] for i in range(window, len(values) + 1)])
    idx = features.index[window - 1 :]
    return sequences, idx


class TorchSignalModel:
    """Wraps GRUClassifier with the same fit/predict_proba/save/load shape
    as SklearnSignalModel, plus the sequence-windowing features.py doesn't
    know about (that's this model's concern, not the feature table's)."""

    def __init__(self, window: int = SEQUENCE_WINDOW, hidden_size: int = 32, epochs: int = 30, lr: float = 1e-3):
        self.window = window
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.net: Optional[GRUClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TorchSignalModel":
        sequences, idx = _make_sequences(X, self.window)
        labels = y.reindex(idx).to_numpy(dtype=np.float32)

        # Seeded init: an unseeded GRU trains to a different model every
        # run, which made train.py's model selection (and therefore the
        # retrain workflow's gate outcome) nondeterministic run to run.
        torch.manual_seed(42)
        self.net = GRUClassifier(n_features=len(FEATURE_COLUMNS), hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        X_t = torch.from_numpy(sequences)
        y_t = torch.from_numpy(labels)

        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.net(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns one probability per row of X, aligned to X's index. Rows
        before the first full window (no `window` days of history yet)
        get 0.5 (neutral / no signal) rather than being dropped, so the
        caller doesn't have to special-case a shorter output length."""
        if self.net is None:
            raise RuntimeError("TorchSignalModel.fit() must be called before predict_proba()")
        out = pd.Series(0.5, index=X.index)
        if len(X) < self.window:
            # Not enough rows to form even one sequence yet -- neutral for
            # everything, same convention as the "before first window"
            # rows below, rather than crashing on an empty stack.
            return out.to_numpy()

        sequences, idx = _make_sequences(X, self.window)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(sequences))
            proba = torch.sigmoid(logits).numpy()

        out.loc[idx] = proba
        return out.to_numpy()

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "window": self.window,
                "hidden_size": self.hidden_size,
                "n_features": len(FEATURE_COLUMNS),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TorchSignalModel":
        checkpoint = torch.load(path, weights_only=True)
        model = cls(window=checkpoint["window"], hidden_size=checkpoint["hidden_size"])
        model.net = GRUClassifier(n_features=checkpoint["n_features"], hidden_size=checkpoint["hidden_size"])
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        return model
