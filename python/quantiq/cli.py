"""End-to-end demo: a baseline technical strategy vs a RAG-sentiment-overlay
strategy vs the trained ML signal (see ml/train.py), backtested on
synthetic price data by default -- or real historical prices with
--real-data (no account/API key needed, see data.py::load_real_ohlcv).

    python -m quantiq.cli
    python -m quantiq.cli --sentiment-backend llm        # score docs with a local LLM instead of the lexicon backend
    python -m quantiq.cli --sentiment-backend finetuned  # score docs with the fine-tuned DistilBERT+LoRA model
    python -m quantiq.cli --real-data --ticker AAPL      # backtest against real Yahoo Finance history
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import generate_synthetic_ohlcv, load_real_ohlcv
from .engine import run_backtest
from .metrics import monte_carlo_var, summarize
from .ml.registry import ModelNotTrainedError, load_metadata
from .rag.retriever import load_corpus
from .rag.signal import build_signal
from .risk import RiskLimits, apply_risk_limits
from .strategies import MovingAverageCrossover, SignalOverlayStrategy
from .volatility import fit_garch, naive_rolling_vol
from .walk_forward import run_walk_forward, summarize_walk_forward

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_docs"


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantIQ backtest demo")
    parser.add_argument("--ticker", default="ACME")
    parser.add_argument(
        "--sentiment-backend",
        choices=["lexicon", "llm", "finetuned"],
        default="lexicon",
        help="lexicon (default, deterministic), llm (Ollama API call), or finetuned (local DistilBERT+LoRA)",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="backtest against real Yahoo Finance history for --ticker instead of synthetic prices",
    )
    parser.add_argument("--period", default="3y", help="yfinance history window when --real-data is set")
    args = parser.parse_args()

    if args.real_data:
        prices = load_real_ohlcv(args.ticker, period=args.period)
        print(f"Using REAL market data: {args.ticker}, {args.period} ({len(prices)} trading days)\n")
    else:
        prices = generate_synthetic_ohlcv(seed=7)

    docs = load_corpus(CORPUS_DIR)
    # The sample RAG corpus (data/sample_docs/) only covers the fictional
    # ticker ACME -- with --real-data on a real ticker, build_signal will
    # correctly find zero matching docs and the sentiment series comes back
    # empty, which SignalOverlayStrategy already handles (missing dates
    # fill to a neutral 0.0 signal). The overlay strategy still runs, it
    # just reduces to (1 - weight) * baseline with no real sentiment tilt --
    # reported plainly below rather than silently, since a real corpus
    # keyed to the requested ticker is a separate piece of work (see
    # README roadmap), not something to fake.
    sentiment = build_signal(docs, tickers=[args.ticker], backend=args.sentiment_backend)
    if args.real_data and sentiment.empty:
        print(
            f"Note: the sample RAG corpus has no documents for {args.ticker!r} (it only covers the "
            "fictional ACME ticker), so the sentiment signal is neutral (all zeros) below -- the "
            "RAG-overlay numbers reduce to a discount of the baseline, not a real sentiment tilt.\n"
        )

    baseline = MovingAverageCrossover()
    overlay = SignalOverlayStrategy(base=MovingAverageCrossover(), signal=sentiment, weight=0.4)

    baseline_result = run_backtest(prices, baseline)
    overlay_result = run_backtest(prices, overlay)

    print(f"Retrieved {len(docs)} sample docs, {sentiment.shape[0]} scored days for {args.ticker}\n")
    print("Baseline (MA crossover):      ", summarize(baseline_result.equity_curve, baseline_result.returns))
    print("RAG-overlay (MA + sentiment): ", summarize(overlay_result.equity_curve, overlay_result.returns))

    try:
        ml_metadata = load_metadata()
        # Deliberately NOT re-running the model against `prices` here: this
        # demo's synthetic series shares its RNG seed with train.py's
        # default training data, so the first several hundred days are
        # bit-identical to data the model trained on -- re-scoring against
        # them would silently show in-sample performance dressed up as a
        # real result. What's actually meaningful is the TRUE chronological
        # held-out performance train.py already computed and recorded at
        # training time (see ml/train.py's evaluate_on_backtest), which is
        # what's printed below instead.
        print(
            f"ML signal (trained {ml_metadata['model_type']}, held-out AUC "
            f"{ml_metadata['held_out_auc']:.3f}, trained on {ml_metadata['data_source']}):",
            ml_metadata["held_out_backtest"],
        )
    except ModelNotTrainedError:
        print(
            "\n(Skipping ML signal: no trained model yet. Run `python -m quantiq.ml.train` "
            "-- optionally with --real-data --ticker <TICKER> -- from python/ first.)"
        )

    wf = run_walk_forward(prices, MovingAverageCrossover(), n_folds=5)
    print("\nWalk-forward (5 folds, MA crossover):", summarize_walk_forward(wf))

    mc = monte_carlo_var(overlay_result.returns, horizon_days=10, confidence=0.95)
    print(f"10-day 95% VaR/CVaR (RAG-overlay, {mc['n_sims']} bootstrap sims):", mc)

    garch_returns = prices["close"].pct_change().dropna()
    garch = fit_garch(garch_returns)
    naive = naive_rolling_vol(garch_returns, window=20)
    print(
        f"\nGARCH(1,1) next-day vol forecast: {garch.forecast_vol:.4f}  "
        f"vs naive 20d rolling vol: {naive:.4f}"
    )

    # apply_risk_limits needs returns aligned 1:1 with positions.index (the
    # full price history), unlike garch_returns above which drops the
    # leading NaN from pct_change() and so is one row shorter.
    asset_returns = prices["close"].pct_change().fillna(0.0)
    risk_limits = RiskLimits(max_position=1.0, max_drawdown=0.10)
    risk_result = apply_risk_limits(baseline_result.positions, asset_returns, risk_limits)
    if risk_result.breach_type:
        print(
            f"\nRisk limits (max_drawdown={risk_limits.max_drawdown:.0%}) on baseline strategy: "
            f"BREACHED ({risk_result.breach_type}) on {risk_result.breach_date.date()}, "
            f"trading halted for the rest of the run. "
            f"Final equity with limits: {risk_result.equity_curve.iloc[-1]:.4f} "
            f"vs unlimited: {baseline_result.equity_curve.iloc[-1]:.4f}"
        )
    else:
        print(f"\nRisk limits (max_drawdown={risk_limits.max_drawdown:.0%}): not breached.")


if __name__ == "__main__":
    main()
