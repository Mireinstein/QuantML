"""Rebalances a real Alpaca PAPER account (no real capital) to match a
strategy's target position, computed from REAL market data.

    python -m quantiq.paper_runner --ticker AAPL --qty-per-unit 10
    python -m quantiq.paper_runner --ticker AAPL --dry-run   # compute only, submit nothing

This is a single-shot rebalance, not a scheduler/daemon -- run it manually,
or point cron/a scheduled task at it if you want periodic rebalancing.
Each run: pulls the latest real price history for `ticker` (data.py's
load_real_ohlcv), computes the strategy's target position in [-1, 1] for
the most recent bar, converts that to a target share count
(target_position * qty_per_unit), compares it against the account's
current REAL (paper) position, and submits whatever market order closes
the gap.
"""
from __future__ import annotations

import argparse

from .data import load_real_ohlcv
from .ml.registry import ModelNotTrainedError, load_best_model
from .paper_trading import PaperTradingError, OrderResult, get_account, get_position, submit_market_order
from .strategies import MeanReversion, MLSignalStrategy, MovingAverageCrossover, Strategy

# Zero-arg-constructible strategies. "ml_signal" is handled separately in
# main() since it needs a loaded model, not just a bare class.
STRATEGIES: dict[str, type[Strategy]] = {
    "ma_crossover": MovingAverageCrossover,
    "mean_reversion": MeanReversion,
}
ML_STRATEGY_NAME = "ml_signal"


def compute_target_shares(
    ticker: str, strategy: Strategy, qty_per_unit: int, period: str = "1y"
) -> tuple[int, float]:
    """Returns (target_share_count, latest_close). Position sizing here is
    intentionally simple (position_fraction * a fixed share count) -- this
    is a demo of wiring real data -> strategy -> real paper order, not a
    portfolio-sizing/risk-budgeting system."""
    prices = load_real_ohlcv(ticker, period=period)
    positions = strategy.positions(prices)
    target_position = float(positions.iloc[-1])  # in [-1, 1], already lookahead-safe (see strategies.py)
    target_shares = round(target_position * qty_per_unit)
    return target_shares, float(prices["close"].iloc[-1])


def rebalance(
    ticker: str, strategy: Strategy, qty_per_unit: int, period: str = "1y", dry_run: bool = False
) -> dict:
    target_shares, last_close = compute_target_shares(ticker, strategy, qty_per_unit, period)

    current_position = get_position(ticker)
    current_shares = int(float(current_position.qty)) if current_position else 0

    delta = target_shares - current_shares
    result: dict = {
        "ticker": ticker,
        "last_close": last_close,
        "current_shares": current_shares,
        "target_shares": target_shares,
        "delta": delta,
        "order": None,
    }
    if delta == 0:
        return result

    side = "buy" if delta > 0 else "sell"
    if dry_run:
        result["order"] = f"DRY RUN -- would {side} {abs(delta)} shares of {ticker}, nothing submitted"
        return result

    order: OrderResult = submit_market_order(ticker, qty=abs(delta), side=side)
    result["order"] = order
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebalance a real Alpaca PAPER account to a strategy's target position, from real market data."
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--strategy", choices=sorted(STRATEGIES) + [ML_STRATEGY_NAME], default="ma_crossover"
    )
    parser.add_argument(
        "--qty-per-unit", type=int, default=10, help="shares representing a full [-1, 1] target position"
    )
    parser.add_argument("--period", default="1y", help="yfinance history window, e.g. 6mo, 1y, 3y")
    parser.add_argument(
        "--dry-run", action="store_true", help="compute and print the rebalance without submitting an order"
    )
    args = parser.parse_args()

    if args.strategy == ML_STRATEGY_NAME:
        try:
            strategy = MLSignalStrategy(model=load_best_model())
        except ModelNotTrainedError as e:
            print(f"Can't use --strategy {ML_STRATEGY_NAME}: {e}")
            return 1
    else:
        strategy = STRATEGIES[args.strategy]()

    try:
        account = get_account()
    except PaperTradingError as e:
        print(f"Could not reach the Alpaca paper API: {e}")
        return 1

    print(f"Paper account: cash=${account.cash:,.2f}  equity=${account.equity:,.2f}  (simulated, not real money)")

    try:
        result = rebalance(args.ticker, strategy, args.qty_per_unit, args.period, args.dry_run)
    except PaperTradingError as e:
        print(f"Rebalance failed: {e}")
        return 1

    print(f"\n{result['ticker']} @ ${result['last_close']:.2f} (real, live price)")
    print(f"Current position: {result['current_shares']} shares")
    print(f"Strategy target:  {result['target_shares']} shares ({strategy.name})")
    if result["delta"] == 0:
        print("Already at target -- no order needed.")
    else:
        print(f"Order: {result['order']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
