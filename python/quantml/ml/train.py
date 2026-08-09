"""Train the ML trading signal.

    python -m quantml.ml.train
    python -m quantml.ml.train --real-data --ticker AAPL --period 5y

Trains a logistic regression baseline, a gradient-boosted tree, and a small
GRU sequence model (model.py) on the SAME chronologically-ordered
train/test split -- unlike TenantIQ's risk model (applicants are
independent rows, so a random stratified split is fine), price history is
sequential, so a random split would leak future information into training
via overlapping rolling-window features. The split here is a single cut
point: everything before it is train, everything after is test, no
shuffling.

Selection isn't just ROC-AUC on the held-out labels -- a model can have a
mediocre AUC and still be a fine trading signal, or a great AUC and be a
bad one (e.g. only right about small, costly-to-trade moves). So each
model's predictions are also run through the real backtester
(MLSignalStrategy + engine.run_backtest) on the same held-out period, and
Sharpe on that held-out backtest is what actually picks the winner.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score

from ..data import generate_synthetic_ohlcv, load_real_ohlcv
from ..engine import run_backtest
from ..metrics import summarize
from ..strategies import MLSignalStrategy
from .features import FEATURE_COLUMNS, LABEL_COLUMN, build_features_and_labels
from .model import SklearnSignalModel, TorchSignalModel, build_gradient_boosting, build_logistic_baseline

HERE = Path(__file__).resolve().parent
METADATA_PATH = HERE / "model_metadata.json"
mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
mlflow.set_experiment("quantml-ml-signal")

TEST_FRACTION = 0.25


def next_version() -> int:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text())["version"] + 1
        except (KeyError, json.JSONDecodeError):
            pass
    return 1


def evaluate_on_backtest(model, prices, test_start_date, run_name: str, params: dict) -> dict:
    """Runs the model's signal through the real backtester on the held-out
    period and logs everything (classification AUC + trading metrics) to
    MLflow under one run."""
    strategy = MLSignalStrategy(model=model, name=run_name)
    result = run_backtest(prices, strategy)
    held_out = result.equity_curve.loc[test_start_date:]
    held_out_returns = result.returns.loc[test_start_date:]
    # Renormalize so the held-out equity curve starts at 1.0 -- otherwise
    # its level would still reflect compounding from the training period,
    # which has nothing to do with this model's held-out performance.
    held_out_equity = held_out / held_out.iloc[0]
    backtest_summary = summarize(held_out_equity, held_out_returns)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({f"backtest_{k}": v for k, v in backtest_summary.items()})

    return backtest_summary


def train_candidates(prices) -> tuple[dict, "pd.Timestamp"]:
    """Trains all three model families on the same chronological split of
    `prices` and returns ({model_name: (model, auc, backtest_summary)},
    test_start_date) -- the pure "given data, produce candidates" step,
    with no argument parsing or disk writes, so it's reusable by both the
    CLI (below) and the autonomous continuous-learning loop
    (autonomous.py), which retrains on an expanding window without
    shelling back out to this script."""
    df = build_features_and_labels(prices)
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    test_start_date = test_df.index[0]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[LABEL_COLUMN]

    candidates = {}

    logistic = SklearnSignalModel(build_logistic_baseline()).fit(X_train, y_train)
    logistic_auc = roc_auc_score(y_test, logistic.predict_proba(X_test))
    logistic_bt = evaluate_on_backtest(
        logistic, prices, test_start_date, "logistic_regression", {"model": "logistic_regression"}
    )
    candidates["logistic_regression"] = (logistic, logistic_auc, logistic_bt)

    gbm = SklearnSignalModel(build_gradient_boosting()).fit(X_train, y_train)
    gbm_auc = roc_auc_score(y_test, gbm.predict_proba(X_test))
    gbm_bt = evaluate_on_backtest(
        gbm, prices, test_start_date, "gradient_boosting", {"model": "gradient_boosting", "max_iter": 200}
    )
    candidates["gradient_boosting"] = (gbm, gbm_auc, gbm_bt)

    gru = TorchSignalModel().fit(X_train, y_train)
    gru_auc = roc_auc_score(y_test, gru.predict_proba(X_test))
    gru_bt = evaluate_on_backtest(
        gru, prices, test_start_date, "gru", {"model": "gru", "window": gru.window, "epochs": gru.epochs}
    )
    candidates["gru"] = (gru, gru_auc, gru_bt)

    return candidates, test_start_date


def select_best(candidates: dict) -> tuple[str, object, float, dict]:
    # Selection metric: held-out Sharpe, not AUC -- see module docstring.
    best_name = max(candidates, key=lambda name: candidates[name][2]["sharpe"])
    best_model, best_auc, best_bt = candidates[best_name]
    return best_name, best_model, best_auc, best_bt


def save_model(best_name: str, best_model, best_auc: float, best_bt: dict, test_start_date, data_source: str) -> dict:
    """Persists the winning model + metadata as the LIVE model everything
    else (`registry.load_best_model()`) reads. This is the "promote" step
    -- callers that want to gate a candidate first (autonomous.py) should
    evaluate it via ml/eval_harness.py *before* calling this."""
    if best_name == "gru":
        best_model.save(HERE / "model.pt")
        model_file = "model.pt"
    else:
        best_model.save(HERE / "model.joblib")
        model_file = "model.joblib"

    metadata = {
        "version": next_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": best_name,
        "model_file": model_file,
        "held_out_auc": best_auc,
        "held_out_backtest": best_bt,
        "test_start_date": str(test_start_date.date()) if hasattr(test_start_date, "date") else str(test_start_date),
        "data_source": data_source,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    return metadata


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the QuantML ML trading signal")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--ticker", default="ACME")
    parser.add_argument("--period", default="5y", help="yfinance window when --real-data is set")
    args = parser.parse_args()

    if args.real_data:
        prices = load_real_ohlcv(args.ticker, period=args.period)
        print(f"Training on REAL market data: {args.ticker}, {args.period} ({len(prices)} trading days)")
    else:
        prices = generate_synthetic_ohlcv(n_days=1500, seed=7)
        print(f"Training on synthetic data ({len(prices)} trading days)")

    candidates, test_start_date = train_candidates(prices)
    for name, (_, auc, bt) in candidates.items():
        print(f"\n{name}: held-out AUC {auc:.3f}")
        print(f"  Held-out backtest: {bt}")

    best_name, best_model, best_auc, best_bt = select_best(candidates)
    print(f"\nSelected: {best_name} (held-out Sharpe {best_bt['sharpe']}, AUC {best_auc:.3f})")

    data_source = f"real:{args.ticker}:{args.period}" if args.real_data else "synthetic"
    metadata = save_model(best_name, best_model, best_auc, best_bt, test_start_date, data_source)
    print(f"\nSaved {metadata['model_file']} (version {metadata['version']})")


if __name__ == "__main__":
    main()
