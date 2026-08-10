"""Live daily paper-trading loop for the ML signal model.

    python -m quantml.autonomous --ticker AAPL

Once per real trading day: pulls TODAY's real market data, asks the
CURRENT live model for its prediction, and rebalances the paper account
to match -- the exact same mechanics as `paper_runner.py`'s single-shot
script, just run continuously so it doesn't need a human to trigger it
every day. Checks periodically (`--check-interval-seconds`, default 1
hour) and simply does nothing if the latest available bar's date hasn't
changed since the last trade -- a real daily bar only updates once a
day, so there's nothing new to act on between real trading days.

Retraining deliberately does NOT happen in this loop -- see
`.github/workflows/retrain-eval.yml`, which retrains on a schedule using
whatever real data has accumulated since the last run, gated through the
same `ml/eval_harness.py` check, and is a genuinely separate concern
from placing today's trade.

An earlier version of this file replayed historical data at an
accelerated pace instead of waiting on real days, retraining every N
replayed cycles. That doesn't hold together: it executed a real order at
TODAY's real market price using a signal computed from a replayed
historical day -- not a backtest (wrong execution price for that day)
and not real trading (wrong signal date for today's price). Removed;
this file now only ever acts on real, current data.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from .data import load_real_ohlcv
from .ml.features import FEATURE_COLUMNS, build_features
from .ml.model import SEQUENCE_WINDOW
from .ml.registry import ModelNotTrainedError, load_best_model, load_metadata
from .ml.train import HERE as ML_DIR
from .paper_trading import PaperTradingError, get_position, submit_market_order

LOG_PATH = ML_DIR / "autonomous_log.jsonl"
STATE_PATH = ML_DIR / "autonomous_state.json"
MAX_LOG_LINES_RETURNED = 200
DEFAULT_CHECK_INTERVAL_SECONDS = 3600


class RunControl:
    """Thread-safe pause/resume flag for `run()`. Optional -- plain CLI
    use (`python -m quantml.autonomous`) never needs one. Exists so
    trader_service.py can run the loop in a background thread while a
    tiny FastAPI app in the same process exposes /pause and /resume for
    the dashboard's Start/Stop buttons to call."""

    def __init__(self, paused: bool = False):
        self._lock = threading.Lock()
        self._paused = paused

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        with self._lock:
            self._paused = value


def _log(event: dict) -> dict:
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    print(json.dumps(event, default=str), flush=True)
    return event


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_traded_date": None}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def recent_activity(n: int = 50) -> list[dict]:
    """Reads the last `n` log lines -- used by the dashboard's activity
    endpoint. Returns an empty list (not an error) if the loop has never
    run, same graceful-degradation convention as the rest of the
    dashboard."""
    if not LOG_PATH.exists():
        return []
    n = min(n, MAX_LOG_LINES_RETURNED)
    lines = LOG_PATH.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]


def feature_drift(row: pd.Series, reference_df: pd.DataFrame, z_threshold: float = 3.0) -> dict:
    """Same idea as TenantIQ's ml/serve.py monitoring: for each feature,
    how many standard deviations is this row from `reference_df`'s mean?
    Flags any feature past `z_threshold`."""
    means = reference_df[FEATURE_COLUMNS].mean()
    stds = reference_df[FEATURE_COLUMNS].std().replace(0, 1.0)
    z_scores = ((row[FEATURE_COLUMNS] - means) / stds).abs()
    drifted = z_scores[z_scores > z_threshold]
    return {
        "max_z_score": float(z_scores.max()),
        "drifted_features": {k: float(v) for k, v in drifted.items()},
    }


def run(
    ticker: str = "AAPL",
    qty_per_unit: int = 10,
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
    max_checks: int | None = None,
    dry_run: bool = False,
    control: RunControl | None = None,
) -> None:
    _log(
        {
            "event": "loop_started",
            "ticker": ticker,
            "check_interval_seconds": check_interval_seconds,
            "dry_run": dry_run,
        }
    )

    # Reference distribution for drift checks: the model's own recorded
    # training-period feature stats, computed once here (not on every
    # check) and reused for the life of this run.
    try:
        metadata = load_metadata()
        if metadata["data_source"] == "synthetic":
            reference_prices = load_real_ohlcv(ticker, period="1y")
        else:
            _, ref_ticker, ref_period = metadata["data_source"].split(":")
            reference_prices = load_real_ohlcv(ref_ticker, period=ref_period)
        reference_features = build_features(reference_prices).loc[: metadata["test_start_date"]]
    except (ModelNotTrainedError, ValueError):
        reference_features = None

    state = _load_state()
    was_paused = False
    checks = 0

    while max_checks is None or checks < max_checks:
        checks += 1
        if control is not None and control.paused:
            if not was_paused:
                _log({"event": "loop_paused"})
                was_paused = True
            time.sleep(check_interval_seconds)
            continue
        if was_paused:
            _log({"event": "loop_resumed"})
            was_paused = False

        try:
            model = load_best_model()
            prices = load_real_ohlcv(ticker, period="1y")
            features = build_features(prices)
            if len(features) < SEQUENCE_WINDOW:
                _log({"event": "insufficient_history", "available_days": len(features)})
                time.sleep(check_interval_seconds)
                continue

            latest_date = features.index[-1]
            latest_date_str = str(latest_date.date())

            if latest_date_str == state.get("last_traded_date"):
                _log({"event": "no_new_trading_day", "latest_date": latest_date_str})
                time.sleep(check_interval_seconds)
                continue

            proba_up = float(model.predict_proba(features)[-1])
            position = max(-1.0, min(1.0, 2 * proba_up - 1))
            target_shares = round(position * qty_per_unit)

            order_result = None
            if dry_run:
                order_result = "DRY RUN -- no order submitted"
            else:
                current = get_position(ticker)
                current_shares = int(float(current.qty)) if current else 0
                delta = target_shares - current_shares
                if delta != 0:
                    side = "buy" if delta > 0 else "sell"
                    try:
                        order = submit_market_order(ticker, qty=abs(delta), side=side)
                        order_result = {"id": order.id, "side": order.side, "qty": order.qty, "status": order.status}
                    except PaperTradingError as e:
                        # A rejected order (e.g. Alpaca's wash-trade guard)
                        # must not stop `last_traded_date` from being
                        # recorded below -- otherwise the loop would retry
                        # the same rejected order forever on this same real
                        # day instead of just waiting for the next one.
                        order_result = {"error": str(e)}

            drift = (
                feature_drift(features.iloc[-1], reference_features)
                if reference_features is not None and len(reference_features)
                else {"max_z_score": None, "drifted_features": {}}
            )

            state["last_traded_date"] = latest_date_str
            _save_state(state)

            _log(
                {
                    "event": "trade",
                    "ticker": ticker,
                    "date": latest_date_str,
                    "predicted_proba_up": proba_up,
                    "suggested_position": position,
                    "target_shares": target_shares,
                    "order": order_result,
                    "drift": drift,
                }
            )

        except Exception as e:  # noqa: BLE001 -- one bad check must not kill the loop
            _log({"event": "check_error", "error": str(e), "traceback": traceback.format_exc()})

        time.sleep(check_interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live daily paper-trading loop for the ML signal model")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--qty-per-unit", type=int, default=10)
    parser.add_argument(
        "--check-interval-seconds",
        type=int,
        default=DEFAULT_CHECK_INTERVAL_SECONDS,
        help="how often to check whether a new real trading day's data is available",
    )
    parser.add_argument("--max-checks", type=int, default=None, help="stop after N checks (default: run forever)")
    parser.add_argument("--dry-run", action="store_true", help="compute everything, submit no real paper orders")
    args = parser.parse_args()

    try:
        run(
            ticker=args.ticker,
            qty_per_unit=args.qty_per_unit,
            check_interval_seconds=args.check_interval_seconds,
            max_checks=args.max_checks,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        _log({"event": "loop_stopped", "reason": "keyboard_interrupt"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
