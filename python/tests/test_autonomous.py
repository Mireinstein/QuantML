import json

import pandas as pd
import pytest

from quantml import autonomous
from quantml.data import generate_synthetic_ohlcv
from quantml.ml.features import FEATURE_COLUMNS, LABEL_COLUMN, build_features_and_labels
from quantml.ml.model import SklearnSignalModel, build_logistic_baseline
from quantml.paper_trading import OrderResult


def test_feature_drift_flags_a_far_outlier_row():
    prices = generate_synthetic_ohlcv(n_days=300, seed=1)
    df = build_features_and_labels(prices)
    train_df = df.iloc[:200]

    normal_row = df.iloc[210]
    drift_normal = autonomous.feature_drift(normal_row, train_df)
    assert drift_normal["drifted_features"] == {}

    outlier_row = normal_row.copy()
    outlier_row["rsi_14"] = train_df["rsi_14"].mean() + 50 * train_df["rsi_14"].std()
    drift_outlier = autonomous.feature_drift(outlier_row, train_df)
    assert "rsi_14" in drift_outlier["drifted_features"]
    assert drift_outlier["max_z_score"] > 3.0


def test_recent_activity_returns_empty_list_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomous, "LOG_PATH", tmp_path / "does_not_exist.jsonl")
    assert autonomous.recent_activity() == []


def test_recent_activity_reads_last_n_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "log.jsonl"
    monkeypatch.setattr(autonomous, "LOG_PATH", log_path)
    with log_path.open("w") as f:
        for i in range(5):
            f.write(json.dumps({"cycle": i}) + "\n")

    activity = autonomous.recent_activity(n=2)
    assert [a["cycle"] for a in activity] == [3, 4]


def test_run_dry_run_never_submits_orders_and_logs_every_cycle(tmp_path, monkeypatch):
    """End-to-end smoke test of the loop's core cycle logic, with the
    filesystem (model registry, log, state) and all outside services
    (Yahoo Finance, Alpaca) fully isolated -- this project's models and
    registry contract are what's under test, not real network calls."""
    prices = generate_synthetic_ohlcv(n_days=400, seed=3)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    df = build_features_and_labels(prices)
    split_idx = int(len(df) * 0.7)
    test_start_date = df.index[split_idx]

    model = SklearnSignalModel(build_logistic_baseline()).fit(
        df[FEATURE_COLUMNS].iloc[:split_idx], df[LABEL_COLUMN].iloc[:split_idx]
    )
    model.save(model_path)
    metadata = {
        "version": 1,
        "model_type": "logistic_regression",
        "model_file": "model.joblib",
        "held_out_auc": 0.55,
        "held_out_backtest": {"sharpe": 0.5},
        "test_start_date": str(test_start_date.date()),
        "data_source": "synthetic",
    }
    metadata_path.write_text(json.dumps(metadata))

    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period: prices)
    monkeypatch.setattr(autonomous, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setattr(autonomous, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(autonomous.eval_harness, "BASELINE_PATH", tmp_path / "eval_baseline.json")

    import quantml.ml.registry as registry_module

    monkeypatch.setattr(registry_module, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(registry_module, "HERE", tmp_path)

    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("submit_market_order must never be called in --dry-run")

    monkeypatch.setattr(autonomous, "submit_market_order", fail_if_called)
    monkeypatch.setattr(autonomous, "get_position", lambda ticker: None)  # noqa: ARG005

    autonomous.run(ticker="TEST", cycle_seconds=0, retrain_every=100, max_cycles=3, dry_run=True)

    activity = autonomous.recent_activity()
    cycle_events = [a for a in activity if a["event"] == "cycle"]
    assert len(cycle_events) == 3
    assert all(e["order"] == "DRY RUN -- no order submitted" for e in cycle_events)
    assert all("predicted_proba_up" in e for e in cycle_events)


def test_order_result_shape_matches_paper_trading_contract():
    """Guards the (id, symbol, qty, side, status) fields autonomous.py
    reads off an OrderResult when logging a submitted order."""
    order = OrderResult(id="abc", symbol="AAPL", qty=5, side="buy", status="accepted")
    assert order.id and order.side == "buy" and order.qty == 5
