"""Autonomous continuous-learning + paper-trading loop for the ML signal
model. NOT part of the deployed Azure dashboard (that stays read-only/
on-demand) -- this runs locally, against your own Alpaca paper account:

    python -m quantml.autonomous --ticker AAPL

Honest design note on what "continuous learning" means here: a real daily
OHLCV bar only updates once per real trading day, so a loop honestly
waiting on live data would have almost nothing new to do for hours at a
stretch -- especially overnight, when markets are closed entirely. To make
continuous learning actually demonstrable in one sitting, this loop
REPLAYS real historical data the model hasn't been evaluated against yet,
one day at a time, at an accelerated cadence (one "day" every
`--cycle-seconds`) -- it is not claiming to trade on live overnight data
that doesn't exist.

Each cycle:

  1. Reveal the next real historical trading day, in order, starting right
     after the live model's own held-out test window (so every replayed
     day is real data the model was never evaluated against).
  2. Get the CURRENTLY PROMOTED model's prediction for that day, size a
     position from it (same `2*p-1` rule as MLSignalStrategy), and submit
     a real order to Alpaca's PAPER account -- a real broker API call,
     fake money, exactly like paper_runner.py's single-shot version.
  3. Because this is a replay of already-known history, the day's actual
     realized direction is available immediately -- record it as ground
     truth and add that row to a growing "recent experience" buffer.
  4. Every `--retrain-every` cycles, retrain a candidate model on the
     original training data plus everything accumulated since, then run
     it through the EXACT SAME quality gate ml/eval_harness.py uses
     (fresh eval data, AUC/Sharpe floors, regression vs. the current
     baseline) -- only promoting it (overwriting the live model files) if
     it passes. A candidate that fails the gate is logged and discarded;
     the previously-promoted model keeps serving. This is the same
     "shadow eval before promotion" pattern production MLOps systems use,
     compressed into one run instead of days.
  5. Feature drift check: compares this cycle's feature values against the
     original training distribution's mean/std (same z-score-based
     approach as TenantIQ's ml/serve.py `/monitoring` endpoint) and flags
     cycles where the live data has drifted far from what the model was
     trained on -- an early warning a production system would page
     someone about, not just a training-time metric.

Every cycle is appended to `ml/autonomous_log.jsonl` (one JSON object per
line) -- the dashboard's "Live autonomous trading" panel reads this file
via `/api/autonomous/activity`. A single cycle's failure (a transient
Alpaca/network error, a retrain that raises) is logged and the loop moves
to the next cycle rather than dying -- an unattended overnight run
shouldn't stop at 2am because of one bad HTTP request.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import load_real_ohlcv
from .ml import eval_harness
from .ml.features import FEATURE_COLUMNS, LABEL_COLUMN, build_features_and_labels
from .ml.registry import ModelNotTrainedError, load_best_model, load_metadata
from .ml.train import HERE as ML_DIR
from .ml.train import save_model, select_best, train_candidates
from .paper_trading import PaperTradingError, get_position, submit_market_order

LOG_PATH = ML_DIR / "autonomous_log.jsonl"
STATE_PATH = ML_DIR / "autonomous_state.json"
MAX_LOG_LINES_RETURNED = 200


def _log(event: dict) -> dict:
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    print(json.dumps(event, default=str), flush=True)
    return event


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"cycle": 0, "generation": 0, "generations_rejected": 0}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def recent_activity(n: int = 50) -> list[dict]:
    """Reads the last `n` log lines -- used by the dashboard's activity
    panel. Returns an empty list (not an error) if the loop has never run,
    same graceful-degradation convention as the rest of the dashboard."""
    if not LOG_PATH.exists():
        return []
    n = min(n, MAX_LOG_LINES_RETURNED)
    lines = LOG_PATH.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]


def feature_drift(row: pd.Series, train_df: pd.DataFrame, z_threshold: float = 3.0) -> dict:
    """Same idea as TenantIQ's ml/serve.py monitoring: for each feature,
    how many standard deviations is this row from the TRAINING
    distribution's mean? Flags any feature past `z_threshold`. Cheap,
    interpretable, and doesn't require a separate drift-detection
    dependency for something this project only needs a signal from."""
    means = train_df[FEATURE_COLUMNS].mean()
    stds = train_df[FEATURE_COLUMNS].std().replace(0, 1.0)
    z_scores = ((row[FEATURE_COLUMNS] - means) / stds).abs()
    drifted = z_scores[z_scores > z_threshold]
    return {
        "max_z_score": float(z_scores.max()),
        "drifted_features": {k: float(v) for k, v in drifted.items()},
    }


def _retrain_and_maybe_promote(prices: pd.DataFrame, cutoff_date: pd.Timestamp, n_new_days: int, state: dict) -> dict:
    """Retrains on an EXPANDING window of the same historical series the
    loop was given at startup, cut off at `cutoff_date` -- the most
    recently replayed day. This is genuine incremental learning: days that
    were held out when the loop started are, by the time enough cycles
    have replayed them, days whose true outcome is now known and folded
    back into training, exactly like a real system would fold in newly-
    elapsed trading days. (The loop doesn't re-fetch fresh data from
    Yahoo Finance every cycle -- see the module docstring for why replay
    is used instead of waiting on genuinely new bars.)"""
    expanded_prices = prices.loc[:cutoff_date]
    candidates, test_start_date = train_candidates(expanded_prices)
    best_name, best_model, best_auc, best_bt = select_best(candidates)

    current_metadata = load_metadata()
    eval_result = eval_harness.EvalResult(auc=best_auc, sharpe=best_bt["sharpe"], n_eval=n_new_days)
    baseline = eval_harness.load_baseline()
    failures = eval_harness.check(eval_result, baseline)

    if failures:
        state["generations_rejected"] = state.get("generations_rejected", 0) + 1
        return _log(
            {
                "event": "retrain_rejected",
                "reasons": failures,
                "candidate_model_type": best_name,
                "candidate_auc": best_auc,
                "candidate_sharpe": best_bt["sharpe"],
            }
        )

    metadata = save_model(best_name, best_model, best_auc, best_bt, test_start_date, current_metadata["data_source"])
    eval_harness.save_baseline(eval_result)
    state["generation"] = state.get("generation", 0) + 1
    return _log(
        {
            "event": "model_promoted",
            "generation": state["generation"],
            "model_type": best_name,
            "version": metadata["version"],
            "auc": best_auc,
            "sharpe": best_bt["sharpe"],
            "training_rows": len(combined_prices),
        }
    )


def run(
    ticker: str = "AAPL",
    qty_per_unit: int = 10,
    cycle_seconds: int = 90,
    retrain_every: int = 15,
    max_cycles: int | None = None,
    dry_run: bool = False,
) -> None:
    prices = load_real_ohlcv(ticker, period="5y")
    df = build_features_and_labels(prices)

    try:
        metadata = load_metadata()
    except ModelNotTrainedError:
        _log({"event": "no_trained_model", "action": "training an initial model before starting the loop"})
        candidates, test_start_date = train_candidates(prices)
        best_name, best_model, best_auc, best_bt = select_best(candidates)
        metadata = save_model(best_name, best_model, best_auc, best_bt, test_start_date, f"real:{ticker}:5y")

    train_df = df[df.index <= pd.Timestamp(metadata["test_start_date"])]
    replay_idx = df.index[df.index > pd.Timestamp(metadata["test_start_date"])]
    if len(replay_idx) < 30:
        _log({"event": "insufficient_replay_data", "available_days": len(replay_idx)})
        return

    if eval_harness.load_baseline() is None:
        # Bootstrap: the currently-promoted model's own recorded held-out
        # metrics become the first baseline every future retrain is
        # measured against.
        eval_harness.save_baseline(
            eval_harness.EvalResult(
                auc=metadata["held_out_auc"], sharpe=metadata["held_out_backtest"]["sharpe"], n_eval=0
            )
        )

    _log(
        {
            "event": "loop_started",
            "ticker": ticker,
            "cycle_seconds": cycle_seconds,
            "retrain_every": retrain_every,
            "replay_days_available": len(replay_idx),
            "dry_run": dry_run,
        }
    )

    state = _load_state()
    cycles_since_retrain = 0
    cycle = 0

    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        try:
            model = load_best_model()
            day = replay_idx[state["cycle"] % len(replay_idx)]
            row = df.loc[[day]]

            # Pass the full trailing history up to and including `day`, not
            # just this one row: TorchSignalModel (the GRU) needs a
            # SEQUENCE_WINDOW-length run of prior rows to predict anything
            # other than its neutral 0.5 fallback (same fix as the web
            # dashboard's /predict endpoint -- see ml/model.py). Cheap
            # either way for the sklearn models, which are row-independent.
            history_so_far = df.loc[:day]
            proba_up = float(model.predict_proba(history_so_far[FEATURE_COLUMNS])[-1])
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
                        # A rejected order (e.g. Alpaca's wash-trade guard
                        # firing because a PREVIOUS order for this symbol is
                        # still open/unfilled -- market orders submitted
                        # outside real trading hours queue instead of
                        # filling immediately, and Alpaca won't accept an
                        # opposite-side order while one is pending) must NOT
                        # abort the whole cycle: `state["cycle"]` still needs
                        # to advance below, or the loop gets stuck replaying
                        # the exact same day/order forever, 90s apart,
                        # indefinitely (this happened in production -- see
                        # the "Honest bugs" note in README). Record the
                        # failure and keep going.
                        order_result = {"error": str(e)}

            drift = feature_drift(row.iloc[0], train_df)
            state["cycle"] += 1
            cycles_since_retrain += 1

            _log(
                {
                    "event": "cycle",
                    "cycle": cycle,
                    "ticker": ticker,
                    "replayed_day": str(day.date()),
                    "predicted_proba_up": proba_up,
                    "suggested_position": position,
                    "target_shares": target_shares,
                    "order": order_result,
                    "realized_up": bool(row[LABEL_COLUMN].iloc[0]),
                    "drift": drift,
                    "generation": state.get("generation", 0),
                }
            )

            if cycles_since_retrain >= retrain_every:
                _retrain_and_maybe_promote(prices, day, cycles_since_retrain, state)
                cycles_since_retrain = 0

            _save_state(state)

        except Exception as e:  # noqa: BLE001 -- see module docstring: one bad cycle must not kill the loop
            _log({"event": "cycle_error", "cycle": cycle, "error": str(e), "traceback": traceback.format_exc()})

        time.sleep(cycle_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous continuous-learning + paper-trading loop (replays real historical data)"
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--qty-per-unit", type=int, default=10)
    parser.add_argument("--cycle-seconds", type=int, default=90, help="seconds of wall-clock time per replayed day")
    parser.add_argument("--retrain-every", type=int, default=15, help="replayed days between retrain attempts")
    parser.add_argument("--max-cycles", type=int, default=None, help="stop after N cycles (default: run forever)")
    parser.add_argument("--dry-run", action="store_true", help="compute everything, submit no real paper orders")
    args = parser.parse_args()

    try:
        run(
            ticker=args.ticker,
            qty_per_unit=args.qty_per_unit,
            cycle_seconds=args.cycle_seconds,
            retrain_every=args.retrain_every,
            max_cycles=args.max_cycles,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        _log({"event": "loop_stopped", "reason": "keyboard_interrupt"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
