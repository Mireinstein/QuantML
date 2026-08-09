"""eval_harness.py -- standalone model-quality gate, separate from the
pytest unit tests. Unit tests check the *code* behaves correctly; this
harness checks the *model artifact on disk* is actually good enough to
ship: it re-evaluates the saved model against FRESH data never used during
training, then fails (non-zero exit) if ROC-AUC or held-out backtest
Sharpe fall below fixed floors, or regress past a tolerance versus the
last recorded baseline (eval_baseline.json).

    python -m quantml.ml.eval_harness
    python -m quantml.ml.eval_harness --update-baseline

Honest limitation: for a model trained on real market data, "fresh" data
means re-fetching the same ticker/period from Yahoo Finance -- which, run
on the same day training happened, is nearly the same history (yfinance
doesn't manufacture new trading days on demand). The synthetic-data case
below IS a fully rigorous disjoint check (a different seed the model has
never seen, same idea as TenantIQ's risk-model eval harness); the
real-data case is a genuine gate against a broken/degenerate model, but
only becomes a true walk-forward check once enough real time has passed
for new trading days to exist beyond the training cutoff.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import roc_auc_score

from ..data import generate_synthetic_ohlcv, load_real_ohlcv
from ..engine import run_backtest
from ..metrics import summarize
from ..strategies import MLSignalStrategy
from .features import FEATURE_COLUMNS, LABEL_COLUMN, build_features_and_labels
from .registry import load_best_model, load_metadata

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "eval_baseline.json"

# Absolute floors: below these the model isn't worth shipping regardless of
# what the recorded baseline says. A random coin flip scores AUC 0.5 and
# Sharpe ~0, so these are set just above "does nothing."
MIN_AUC = 0.52
MIN_SHARPE = 0.0
MAX_AUC_REGRESSION = 0.05
MAX_SHARPE_REGRESSION = 0.5


@dataclass
class EvalResult:
    auc: float
    sharpe: float
    n_eval: int


def get_fresh_eval_data(metadata: dict):
    source = metadata["data_source"]
    if source == "synthetic":
        # A seed the training run (seed=7 in train.py) never saw -- fully
        # disjoint, same idea as TenantIQ's eval harness.
        return generate_synthetic_ohlcv(n_days=1500, seed=999)
    _, ticker, period = source.split(":")
    return load_real_ohlcv(ticker, period=period)


def evaluate_model(model, metadata: dict) -> EvalResult:
    prices = get_fresh_eval_data(metadata)
    df = build_features_and_labels(prices)
    X, y = df[FEATURE_COLUMNS], df[LABEL_COLUMN]

    auc = float(roc_auc_score(y, model.predict_proba(X)))

    strategy = MLSignalStrategy(model=model, name="eval")
    result = run_backtest(prices, strategy)
    sharpe = summarize(result.equity_curve, result.returns)["sharpe"]

    return EvalResult(auc=auc, sharpe=sharpe, n_eval=len(df))


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text())


def save_baseline(result: EvalResult) -> None:
    BASELINE_PATH.write_text(json.dumps({"auc": result.auc, "sharpe": result.sharpe}, indent=2))


def check(result: EvalResult, baseline: dict | None) -> list[str]:
    failures = []
    if result.auc < MIN_AUC:
        failures.append(f"AUC {result.auc:.3f} is below the minimum {MIN_AUC}")
    if result.sharpe < MIN_SHARPE:
        failures.append(f"Held-out Sharpe {result.sharpe:.3f} is below the minimum {MIN_SHARPE}")
    if baseline is not None:
        auc_drop = baseline["auc"] - result.auc
        sharpe_drop = baseline["sharpe"] - result.sharpe
        if auc_drop > MAX_AUC_REGRESSION:
            failures.append(f"AUC regressed by {auc_drop:.3f} versus baseline {baseline['auc']:.3f}")
        if sharpe_drop > MAX_SHARPE_REGRESSION:
            failures.append(f"Sharpe regressed by {sharpe_drop:.3f} versus baseline {baseline['sharpe']:.3f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained ML signal model against quality gates")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    try:
        metadata = load_metadata()
        model = load_best_model()
    except Exception as e:  # ModelNotTrainedError or a corrupt artifact
        print(f"Could not load a trained model: {e}")
        return 1

    result = evaluate_model(model, metadata)
    baseline = load_baseline()

    print(f"Model: {metadata['model_type']} (version {metadata['version']})")
    print(f"Eval set: {result.n_eval} fresh rows")
    print(f"AUC: {result.auc:.3f}   Held-out backtest Sharpe: {result.sharpe:.3f}")
    print(f"Baseline: AUC {baseline['auc']:.3f}, Sharpe {baseline['sharpe']:.3f}" if baseline else "No baseline yet.")

    failures = check(result, baseline)
    if failures:
        print("\nFAILED quality gate:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASSED quality gate.")
    if args.update_baseline:
        save_baseline(result)
        print(f"Updated baseline at {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
