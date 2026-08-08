"""Order-book-driven execution simulation.

Feeds synthetic background liquidity into the REAL C++ OrderBook (via the
quantiq_cpp pybind11 module) tick by tick, then submits a strategy's order
at each bar boundary and records the actual fill(s) it receives -- including
slippage from crossing the book -- instead of assuming a fill at an
idealized mid price. This is what actually changes when a strategy is
executed by a matching engine rather than approximated by
`engine.run_backtest`'s vectorized position * returns model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import quantiq_cpp as qc


@dataclass
class ExecutionResult:
    bar_index: list = field(default_factory=list)
    target_delta: list = field(default_factory=list)      # requested change in position (signed)
    filled_delta: list = field(default_factory=list)      # actually filled change (signed)
    avg_fill_price: list = field(default_factory=list)
    mid_price_at_submit: list = field(default_factory=list)
    slippage_bps: list = field(default_factory=list)      # positive = worse than mid, for either side


def simulate_execution(
    ticks: np.ndarray,
    target_positions: np.ndarray,
    ticks_per_bar: int = 50,
    order_notional_qty: int = 100,
    background_order_qty: int = 150,
    spread_ticks: int = 2,
) -> ExecutionResult:
    """Runs a synthetic order-book execution simulation.

    At each tick, a synthetic background participant refreshes its resting
    quote on each side of that tick's mid price (cancelling the previous
    tick's quote first), standing in for the rest of the market continuously
    repricing around the current mid. Without this refresh, liquidity from
    early-bar ticks would sit stale at off-market prices and the strategy's
    order could fill against prices from many ticks ago -- which is what a
    real book does NOT do, since market makers cancel/replace as price moves.

    At each bar boundary, the strategy's target position (in [-1, 1]) is
    translated into a marketable order sized against `order_notional_qty`
    and submitted to the SAME order book, so it fills against real
    (synthetic) resting liquidity rather than an assumed price.
    `background_order_qty` must be >= `order_notional_qty` for a strategy
    order to be able to fill in full against a single tick's liquidity.
    """
    book = qc.OrderBook()
    result = ExecutionResult()

    order_id = 1
    current_qty = 0
    prev_buy_id = None
    prev_sell_id = None

    n_bars = len(target_positions)
    for bar in range(n_bars):
        bar_start = bar * ticks_per_bar
        bar_end = min(bar_start + ticks_per_bar, len(ticks))
        if bar_start >= len(ticks):
            break

        for t in range(bar_start, bar_end):
            mid = int(ticks[t])
            if prev_buy_id is not None:
                book.cancel(prev_buy_id)
                book.cancel(prev_sell_id)
            prev_buy_id = order_id
            book.submit(qc.Order(prev_buy_id, qc.Side.Buy, mid - spread_ticks, background_order_qty))
            order_id += 1
            prev_sell_id = order_id
            book.submit(qc.Order(prev_sell_id, qc.Side.Sell, mid + spread_ticks, background_order_qty))
            order_id += 1

        mid_at_submit = int(ticks[bar_end - 1])
        target_qty = int(round(target_positions[bar] * order_notional_qty))
        delta = target_qty - current_qty
        if delta == 0:
            continue

        side = qc.Side.Buy if delta > 0 else qc.Side.Sell
        # Marketable limit: priced deep enough to guarantee a fill against
        # the resting liquidity just posted, so we measure execution
        # quality (slippage) rather than whether it fills at all.
        limit_price = mid_at_submit + (spread_ticks * 5 if delta > 0 else -spread_ticks * 5)
        fills = book.submit(qc.Order(order_id, side, limit_price, abs(delta)))
        order_id += 1

        filled = sum(f.qty for f in fills)
        notional = sum(f.qty * f.price for f in fills)
        avg_price = (notional / filled) if filled else None
        signed_filled = filled if delta > 0 else -filled
        current_qty += signed_filled

        result.bar_index.append(bar)
        result.target_delta.append(delta)
        result.filled_delta.append(signed_filled)
        result.avg_fill_price.append(avg_price)
        result.mid_price_at_submit.append(mid_at_submit)
        if avg_price is not None:
            raw_slip = (avg_price - mid_at_submit) / mid_at_submit * 10_000
            result.slippage_bps.append(raw_slip if delta > 0 else -raw_slip)
        else:
            result.slippage_bps.append(None)

    return result


def summarize_execution(result: ExecutionResult) -> dict:
    valid_slip = [s for s in result.slippage_bps if s is not None]
    fully_filled = sum(
        1 for f, t in zip(result.filled_delta, result.target_delta) if abs(f) == abs(t)
    )
    n_orders = len(result.target_delta)
    return {
        "orders_submitted": n_orders,
        "fill_rate": round(fully_filled / n_orders, 3) if n_orders else 0.0,
        "avg_slippage_bps": round(sum(valid_slip) / len(valid_slip), 2) if valid_slip else None,
        "max_slippage_bps": round(max(valid_slip), 2) if valid_slip else None,
    }
