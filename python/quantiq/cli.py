"""End-to-end demo: baseline technical strategy vs a RAG-sentiment-overlay
strategy, backtested on synthetic price data.

    python -m quantiq.cli
    python -m quantiq.cli --use-llm   # score docs with a local LLM instead of the lexicon backend
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import generate_synthetic_ohlcv
from .engine import run_backtest
from .metrics import summarize
from .rag.retriever import load_corpus
from .rag.signal import build_signal
from .strategies import MovingAverageCrossover, SignalOverlayStrategy

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_docs"


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantIQ backtest demo")
    parser.add_argument("--ticker", default="ACME")
    parser.add_argument(
        "--use-llm", action="store_true", help="score docs with an LLM instead of the lexicon backend"
    )
    args = parser.parse_args()

    prices = generate_synthetic_ohlcv(seed=7)

    docs = load_corpus(CORPUS_DIR)
    sentiment = build_signal(docs, tickers=[args.ticker], use_llm=args.use_llm)

    baseline = MovingAverageCrossover()
    overlay = SignalOverlayStrategy(base=MovingAverageCrossover(), signal=sentiment, weight=0.4)

    baseline_result = run_backtest(prices, baseline)
    overlay_result = run_backtest(prices, overlay)

    print(f"Retrieved {len(docs)} sample docs, {sentiment.shape[0]} scored days for {args.ticker}\n")
    print("Baseline (MA crossover):      ", summarize(baseline_result.equity_curve, baseline_result.returns))
    print("RAG-overlay (MA + sentiment): ", summarize(overlay_result.equity_curve, overlay_result.returns))


if __name__ == "__main__":
    main()
