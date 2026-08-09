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


def test_run_advances_past_a_rejected_order_instead_of_looping_forever(tmp_path, monkeypatch):
    """Regression test for a real production bug: Alpaca can reject a new
    order (e.g. its wash-trade guard firing because a previous order for
    the same symbol is still open/unfilled -- market orders submitted
    outside real trading hours queue rather than fill immediately). Before
    the fix, that PaperTradingError propagated up and aborted the cycle
    BEFORE `state["cycle"]` was incremented, so the loop retried the exact
    same day/order forever, 90s apart, indefinitely -- confirmed live: 132
    consecutive identical failures over ~3.3 hours. The fix: a rejected
    order is recorded in the log and the cycle still completes normally."""
    prices = generate_synthetic_ohlcv(n_days=400, seed=5)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    df = build_features_and_labels(prices)
    split_idx = int(len(df) * 0.7)
    test_start_date = df.index[split_idx]

    model = SklearnSignalModel(build_logistic_baseline()).fit(
        df[FEATURE_COLUMNS].iloc[:split_idx], df[LABEL_COLUMN].iloc[:split_idx]
    )
    model.save(model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "logistic_regression",
                "model_file": "model.joblib",
                "held_out_auc": 0.55,
                "held_out_backtest": {"sharpe": 0.5},
                "test_start_date": str(test_start_date.date()),
                "data_source": "synthetic",
            }
        )
    )

    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period: prices)
    monkeypatch.setattr(autonomous, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setattr(autonomous, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(autonomous.eval_harness, "BASELINE_PATH", tmp_path / "eval_baseline.json")

    import quantml.ml.registry as registry_module
    from quantml.paper_trading import PaperTradingError, Position

    monkeypatch.setattr(registry_module, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(registry_module, "HERE", tmp_path)

    # A large existing position guarantees a non-zero delta every cycle
    # (target_shares is bounded to +-qty_per_unit), so submit_market_order
    # is always attempted -- and always rejected here, simulating the
    # wash-trade scenario.
    monkeypatch.setattr(
        autonomous,
        "get_position",
        lambda ticker: Position(symbol=ticker, qty=100, avg_entry_price=100.0, market_value=10000.0, unrealized_pl=0.0),  # noqa: ARG005
    )

    def always_reject(*args, **kwargs):  # noqa: ARG001
        raise PaperTradingError("Alpaca paper API returned 403: potential wash trade detected")

    monkeypatch.setattr(autonomous, "submit_market_order", always_reject)

    autonomous.run(ticker="TEST", qty_per_unit=10, cycle_seconds=0, retrain_every=100, max_cycles=5, dry_run=False)

    activity = autonomous.recent_activity()
    cycle_events = [a for a in activity if a["event"] == "cycle"]
    error_events = [a for a in activity if a["event"] == "cycle_error"]

    assert len(cycle_events) == 5  # all 5 cycles completed, none aborted into cycle_error
    assert error_events == []
    assert all(e["order"] == {"error": "Alpaca paper API returned 403: potential wash trade detected"} for e in cycle_events)
    # The critical assertion: each cycle replayed a DIFFERENT day -- state
    # advanced despite every order being rejected, instead of getting
    # stuck replaying the same day forever.
    replayed_days = [e["replayed_day"] for e in cycle_events]
    assert len(set(replayed_days)) == 5


def test_run_control_pause_resume_round_trip():
    control = autonomous.RunControl()
    assert control.paused is False
    control.paused = True
    assert control.paused is True
    control.paused = False
    assert control.paused is False


def test_run_stays_paused_and_logs_nothing_per_cycle_while_paused(tmp_path, monkeypatch):
    """A paused loop still does its one-time setup (load data, ensure a
    model exists) -- that happens once at container boot in
    trader_service.py, not on every start/resume -- but must NEVER run
    the per-cycle trading logic (predicting, placing an order) while
    paused, and must log exactly one "loop_paused" event, not one per
    wake-up. Also regression coverage for the loop_passes/cycle split: a
    paused loop never advances `cycle`, so bounding the while loop on
    `cycle` (instead of `loop_passes`) would spin forever here."""
    prices = generate_synthetic_ohlcv(n_days=400, seed=7)

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    df = build_features_and_labels(prices)
    split_idx = int(len(df) * 0.7)
    test_start_date = df.index[split_idx]
    model = SklearnSignalModel(build_logistic_baseline()).fit(
        df[FEATURE_COLUMNS].iloc[:split_idx], df[LABEL_COLUMN].iloc[:split_idx]
    )
    model.save(model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "logistic_regression",
                "model_file": "model.joblib",
                "held_out_auc": 0.55,
                "held_out_backtest": {"sharpe": 0.5},
                "test_start_date": str(test_start_date.date()),
                "data_source": "synthetic",
            }
        )
    )

    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period: prices)
    monkeypatch.setattr(autonomous, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setattr(autonomous, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(autonomous.eval_harness, "BASELINE_PATH", tmp_path / "eval_baseline.json")

    import quantml.ml.registry as registry_module

    monkeypatch.setattr(registry_module, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(registry_module, "HERE", tmp_path)

    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("must not run per-cycle trading logic while paused")

    monkeypatch.setattr(autonomous, "load_best_model", fail_if_called)
    monkeypatch.setattr(autonomous, "get_position", fail_if_called)
    monkeypatch.setattr(autonomous, "submit_market_order", fail_if_called)

    control = autonomous.RunControl(paused=True)
    autonomous.run(ticker="TEST", cycle_seconds=0, max_cycles=3, control=control)

    activity = autonomous.recent_activity()
    assert [a["event"] for a in activity] == ["loop_started", "loop_paused"]


def test_order_result_shape_matches_paper_trading_contract():
    """Guards the (id, symbol, qty, side, status) fields autonomous.py
    reads off an OrderResult when logging a submitted order."""
    order = OrderResult(id="abc", symbol="AAPL", qty=5, side="buy", status="accepted")
    assert order.id and order.side == "buy" and order.qty == 5
