"""eval_harness.py -- standalone quality gate for the fine-tuned sentiment
model, same shape as ml/eval_harness.py (the trading-signal model) and
TenantIQ's ml/eval_harness.py (the risk model): re-evaluates the saved
adapter against a FRESH slice of the real validation set never scored
during training-time reporting, and fails if accuracy/F1 drop below fixed
floors or regress past a tolerance versus the last recorded baseline.

    python -m quantml.finetune.eval_harness
    python -m quantml.finetune.eval_harness --update-baseline
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from .data import load_sentiment_data
from .model import FineTunedSentimentScorer, ModelNotTrainedError

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "eval_baseline.json"

MIN_ACCURACY = 0.55  # majority-class ("Neutral") baseline is ~0.65 on this dataset's
MIN_F1_MACRO = 0.40  # class distribution, so these floors require real per-class signal
MAX_ACCURACY_REGRESSION = 0.05
MAX_F1_REGRESSION = 0.05


@dataclass
class EvalResult:
    accuracy: float
    f1_macro: float
    n_eval: int


def evaluate_model(scorer: FineTunedSentimentScorer, n_samples: int = 500) -> EvalResult:
    data = load_sentiment_data()
    eval_slice = data.validation.shuffle(seed=999).select(range(min(n_samples, len(data.validation))))

    preds = [scorer.label_names.index(scorer.predict(row["text"])["label"]) for row in eval_slice]
    labels = list(eval_slice["label"])

    return EvalResult(
        accuracy=accuracy_score(labels, preds),
        f1_macro=f1_score(labels, preds, average="macro"),
        n_eval=len(labels),
    )


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text())


def save_baseline(result: EvalResult) -> None:
    BASELINE_PATH.write_text(json.dumps({"accuracy": result.accuracy, "f1_macro": result.f1_macro}, indent=2))


def check(result: EvalResult, baseline: dict | None) -> list[str]:
    failures = []
    if result.accuracy < MIN_ACCURACY:
        failures.append(f"Accuracy {result.accuracy:.3f} is below the minimum {MIN_ACCURACY}")
    if result.f1_macro < MIN_F1_MACRO:
        failures.append(f"F1 (macro) {result.f1_macro:.3f} is below the minimum {MIN_F1_MACRO}")
    if baseline is not None:
        acc_drop = baseline["accuracy"] - result.accuracy
        f1_drop = baseline["f1_macro"] - result.f1_macro
        if acc_drop > MAX_ACCURACY_REGRESSION:
            failures.append(f"Accuracy regressed by {acc_drop:.3f} versus baseline {baseline['accuracy']:.3f}")
        if f1_drop > MAX_F1_REGRESSION:
            failures.append(f"F1 (macro) regressed by {f1_drop:.3f} versus baseline {baseline['f1_macro']:.3f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned sentiment model against quality gates")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    try:
        scorer = FineTunedSentimentScorer()
    except ModelNotTrainedError as e:
        print(f"Could not load a fine-tuned model: {e}")
        return 1

    result = evaluate_model(scorer)
    baseline = load_baseline()

    print(f"Eval set: {result.n_eval} held-out tweets")
    print(f"Accuracy: {result.accuracy:.3f}   F1 (macro): {result.f1_macro:.3f}")
    print(f"Baseline: accuracy {baseline['accuracy']:.3f}, F1 {baseline['f1_macro']:.3f}" if baseline else "No baseline yet.")

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
