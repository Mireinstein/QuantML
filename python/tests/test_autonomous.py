import json

import pandas as pd
import pytest

from quantml import autonomous
from quantml.data import generate_synthetic_ohlcv
from quantml.ml.features import build_features, build_features_and_labels
from quantml.ml.model import SklearnSignalModel, build_logistic_baseline
from quantml.paper_trading import OrderResult, PaperTradingError, Position


def test_feature_drift_flags_a_far_outlier_row():
    prices = generate_synthetic_ohlcv(n_days=300, seed=1)
    df = build_features_and_labels(prices)
    reference_df = df.iloc[:200]

    normal_row = df.iloc[210]
    drift_normal = autonomous.feature_drift(normal_row, reference_df)
    assert drift_normal["drifted_features"] == {}

    outlier_row = normal_row.copy()
    outlier_row["rsi_14"] = reference_df["rsi_14"].mean() + 50 * reference_df["rsi_14"].std()
    drift_outlier = autonomous.feature_drift(outlier_row, reference_df)
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


def test_run_control_pause_resume_round_trip():
    control = autonomous.RunControl()
    assert control.paused is False
    control.paused = True
    assert control.paused is True
    control.paused = False
    assert control.paused is False


@pytest.fixture
def trained_model(tmp_path, monkeypatch):
    """Sets up a real trained model + metadata in an isolated registry
    location, and points autonomous.py's LOG_PATH/STATE_PATH there too.
    Returns the synthetic prices used to train it, so tests can serve the
    same (or different) data back through the mocked load_real_ohlcv."""
    prices = generate_synthetic_ohlcv(n_days=400, seed=3)
    df = build_features_and_labels(prices)
    split_idx = int(len(df) * 0.7)
    test_start_date = df.index[split_idx]

    from quantml.ml.features import FEATURE_COLUMNS, LABEL_COLUMN

    model = SklearnSignalModel(build_logistic_baseline()).fit(
        df[FEATURE_COLUMNS].iloc[:split_idx], df[LABEL_COLUMN].iloc[:split_idx]
    )
    model_path = tmp_path / "model.joblib"
    model.save(model_path)

    metadata_path = tmp_path / "model_metadata.json"
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

    import quantml.ml.registry as registry_module

    monkeypatch.setattr(registry_module, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(registry_module, "HERE", tmp_path)
    monkeypatch.setattr(autonomous, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setattr(autonomous, "STATE_PATH", tmp_path / "state.json")

    return prices


def test_run_trades_on_a_new_real_day_then_stops_repeating_it(trained_model, monkeypatch):
    """The core behavior: given real data with a "new" latest day (state
    has no last_traded_date yet), one check computes a prediction and
    submits an order, records that day as traded, and every subsequent
    check on the SAME data does nothing (logs no_new_trading_day) instead
    of re-trading the same real day over and over."""
    prices = trained_model
    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period="1y": prices)

    orders = []

    def _fake_submit(ticker, qty, side):  # noqa: ARG001
        orders.append((side, qty))
        return OrderResult(id="1", symbol=ticker, qty=qty, side=side, status="accepted")

    monkeypatch.setattr(autonomous, "submit_market_order", _fake_submit)
    monkeypatch.setattr(autonomous, "get_position", lambda ticker: None)  # noqa: ARG005

    autonomous.run(ticker="TEST", check_interval_seconds=0, max_checks=3, dry_run=False)

    activity = autonomous.recent_activity()
    trade_events = [a for a in activity if a["event"] == "trade"]
    skip_events = [a for a in activity if a["event"] == "no_new_trading_day"]

    assert len(trade_events) == 1  # only traded once, not on every check
    assert len(skip_events) == 2  # the other two checks correctly saw nothing new


def test_run_dry_run_never_submits_orders(trained_model, monkeypatch):
    prices = trained_model
    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period="1y": prices)

    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("submit_market_order must never be called in --dry-run")

    monkeypatch.setattr(autonomous, "submit_market_order", fail_if_called)
    monkeypatch.setattr(autonomous, "get_position", lambda ticker: None)  # noqa: ARG005

    autonomous.run(ticker="TEST", check_interval_seconds=0, max_checks=1, dry_run=True)

    activity = autonomous.recent_activity()
    trade_events = [a for a in activity if a["event"] == "trade"]
    assert len(trade_events) == 1
    assert trade_events[0]["order"] == "DRY RUN -- no order submitted"


def test_run_records_the_day_as_traded_even_when_the_order_is_rejected(trained_model, monkeypatch):
    """Regression coverage for the class of bug that broke the old
    replay-based loop: a rejected order must not stop `last_traded_date`
    from being recorded, or the loop would retry the same rejected order
    forever instead of just waiting for the next real trading day."""
    prices = trained_model
    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period="1y": prices)
    monkeypatch.setattr(
        autonomous,
        "get_position",
        lambda ticker: Position(symbol=ticker, qty=100, avg_entry_price=100.0, market_value=10000.0, unrealized_pl=0.0),  # noqa: ARG005
    )

    def always_reject(*args, **kwargs):  # noqa: ARG001
        raise PaperTradingError("Alpaca paper API returned 403: potential wash trade detected")

    monkeypatch.setattr(autonomous, "submit_market_order", always_reject)

    autonomous.run(ticker="TEST", qty_per_unit=10, check_interval_seconds=0, max_checks=2, dry_run=False)

    activity = autonomous.recent_activity()
    trade_events = [a for a in activity if a["event"] == "trade"]
    skip_events = [a for a in activity if a["event"] == "no_new_trading_day"]
    error_events = [a for a in activity if a["event"] == "check_error"]

    assert len(trade_events) == 1
    assert trade_events[0]["order"] == {"error": "Alpaca paper API returned 403: potential wash trade detected"}
    assert error_events == []
    # second check saw the same (still-latest) real day and correctly did nothing
    assert len(skip_events) == 1


def test_run_stays_paused_and_never_calls_trading_logic(trained_model, monkeypatch):
    prices = trained_model
    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period="1y": prices)

    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("must not run trading logic while paused")

    monkeypatch.setattr(autonomous, "load_best_model", fail_if_called)
    monkeypatch.setattr(autonomous, "get_position", fail_if_called)
    monkeypatch.setattr(autonomous, "submit_market_order", fail_if_called)

    control = autonomous.RunControl(paused=True)
    autonomous.run(ticker="TEST", check_interval_seconds=0, max_checks=3, control=control)

    activity = autonomous.recent_activity()
    assert [a["event"] for a in activity] == ["loop_started", "loop_paused"]


def test_order_result_shape_matches_paper_trading_contract():
    """Guards the (id, symbol, qty, side, status) fields autonomous.py
    reads off an OrderResult when logging a submitted order."""
    order = OrderResult(id="abc", symbol="AAPL", qty=5, side="buy", status="accepted")
    assert order.id and order.side == "buy" and order.qty == 5


def test_completed_bars_drops_todays_bar_during_market_hours():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    idx = pd.bdate_range(end="2026-08-18", periods=5)
    features = pd.DataFrame({"x": range(5)}, index=idx)

    mid_session = datetime(2026, 8, 18, 11, 30, tzinfo=ZoneInfo("America/New_York"))
    assert len(autonomous.completed_bars(features, now_ny=mid_session)) == 4

    after_close = datetime(2026, 8, 18, 16, 10, tzinfo=ZoneInfo("America/New_York"))
    assert len(autonomous.completed_bars(features, now_ny=after_close)) == 5

    next_day = datetime(2026, 8, 19, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    assert len(autonomous.completed_bars(features, now_ny=next_day)) == 5


def test_sleep_returns_early_when_pause_flag_flips():
    import threading
    import time as time_module

    control = autonomous.RunControl(paused=False)
    threading.Timer(0.2, lambda: setattr(control, "paused", True)).start()

    start = time_module.monotonic()
    autonomous._sleep(5, control)
    elapsed = time_module.monotonic() - start
    assert elapsed < 3  # returned on the flip, not after the full 5s


def test_day_is_recorded_as_traded_before_drift_check_runs(trained_model, monkeypatch):
    """A failure after the order decision (e.g. in the drift check) must
    not cause the next check to re-run the same day's trade."""
    prices = trained_model
    monkeypatch.setattr(autonomous, "load_real_ohlcv", lambda ticker, period="1y": prices)
    monkeypatch.setattr(autonomous, "get_position", lambda ticker: None)  # noqa: ARG005

    submissions = []

    def _submit(ticker, qty, side):
        submissions.append((side, qty))
        return OrderResult(id="1", symbol=ticker, qty=qty, side=side, status="accepted")

    monkeypatch.setattr(autonomous, "submit_market_order", _submit)

    def _drift_raises(*args, **kwargs):  # noqa: ARG001
        raise ValueError("drift computation blew up")

    monkeypatch.setattr(autonomous, "feature_drift", _drift_raises)

    autonomous.run(ticker="TEST", check_interval_seconds=0, max_checks=2, dry_run=False)

    activity = autonomous.recent_activity()
    assert len(submissions) <= 1  # never re-traded the same day
    assert any(a["event"] == "no_new_trading_day" for a in activity)
