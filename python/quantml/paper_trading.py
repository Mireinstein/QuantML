"""A thin REST client for Alpaca's PAPER trading API -- simulated orders
against real market prices, fake starting cash, zero real capital at risk.

The base URL below is hard-coded to Alpaca's paper subdomain and is
deliberately NOT configurable via an environment variable or constructor
argument: a paper-trading client should never be one misconfigured env var
away from submitting a real order against a live account. If you actually
want live trading, that's a different, more dangerous integration this
project does not provide.

Credentials (free, from https://app.alpaca.markets/paper/dashboard/overview
-- this project doesn't and can't create that account for you). Either
export them in your shell, or -- easier, and the recommended way -- put
them in a `python/.env` file (gitignored, never committed, and never seen
by anyone/anything other than your own filesystem and this process):

    # python/.env
    ALPACA_API_KEY_ID=...
    ALPACA_API_SECRET_KEY=...

Every call raises PaperTradingError with a clear message if the credentials
are missing or Alpaca rejects the request -- this module never silently
no-ops an order. A caller that ignores the exception and assumes an order
went through would be trading on a false belief, which is worse than a
loud failure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import requests
from dotenv import load_dotenv

# Loads python/.env if present; a no-op (and harmless) if it doesn't exist
# or the vars are already set in the real environment -- shell exports
# always win since load_dotenv() doesn't override existing env vars.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Hard-coded on purpose -- see module docstring. Do not make this an env var.
PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class PaperTradingError(RuntimeError):
    pass


@dataclass
class Account:
    cash: float
    portfolio_value: float
    equity: float
    buying_power: float


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float


@dataclass
class OrderResult:
    id: str
    symbol: str
    qty: float
    side: str
    status: str


@dataclass
class OrderRecord:
    id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    submitted_at: str


@dataclass
class PortfolioHistory:
    timestamps: list[str]
    equity: list[float]
    profit_loss: list[float]
    profit_loss_pct: list[float]
    base_value: float


def _credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise PaperTradingError(
            "Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY. Sign up for a free "
            "paper trading account at https://app.alpaca.markets/paper/dashboard/overview, "
            "generate an API key pair there, and export both env vars."
        )
    return key, secret


def _headers() -> dict:
    key, secret = _credentials()
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{PAPER_BASE_URL}{path}"
    try:
        resp = requests.request(method, url, headers=_headers(), timeout=10, **kwargs)
    except requests.RequestException as e:
        raise PaperTradingError(f"Request to Alpaca paper API failed: {e}") from e

    if not resp.ok:
        raise PaperTradingError(f"Alpaca paper API returned {resp.status_code}: {resp.text}")
    return resp.json() if resp.text else {}


def get_account() -> Account:
    data = _request("GET", "/v2/account")
    return Account(
        cash=float(data["cash"]),
        portfolio_value=float(data["portfolio_value"]),
        equity=float(data["equity"]),
        buying_power=float(data["buying_power"]),
    )


def get_position(symbol: str) -> Optional[Position]:
    """Returns None (not an error) if there's simply no open position in
    `symbol` -- that's an expected, common state, not a failure."""
    try:
        data = _request("GET", f"/v2/positions/{symbol}")
    except PaperTradingError as e:
        if "404" in str(e):
            return None
        raise
    return Position(
        symbol=data["symbol"],
        qty=float(data["qty"]),
        avg_entry_price=float(data["avg_entry_price"]),
        market_value=float(data["market_value"]),
        unrealized_pl=float(data["unrealized_pl"]),
    )


def list_positions() -> list[Position]:
    data = _request("GET", "/v2/positions")
    return [
        Position(
            symbol=p["symbol"],
            qty=float(p["qty"]),
            avg_entry_price=float(p["avg_entry_price"]),
            market_value=float(p["market_value"]),
            unrealized_pl=float(p["unrealized_pl"]),
        )
        for p in data
    ]


def submit_market_order(symbol: str, qty: float, side: Literal["buy", "sell"]) -> OrderResult:
    if qty <= 0:
        raise PaperTradingError(f"qty must be positive, got {qty}")
    data = _request(
        "POST",
        "/v2/orders",
        json={
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        },
    )
    return OrderResult(
        id=data["id"], symbol=data["symbol"], qty=float(data["qty"]), side=data["side"], status=data["status"]
    )


def list_orders(status: str = "all", limit: int = 50) -> list[OrderRecord]:
    """Real order history straight from Alpaca -- ground truth for what
    the account has actually done, not a reconstruction from a local log.
    `filled_qty`/`filled_avg_price` distinguish an order that was merely
    accepted from one that actually executed."""
    data = _request("GET", "/v2/orders", params={"status": status, "limit": limit})
    return [
        OrderRecord(
            id=o["id"],
            symbol=o["symbol"],
            side=o["side"],
            qty=float(o["qty"]),
            status=o["status"],
            filled_qty=float(o["filled_qty"]),
            filled_avg_price=float(o["filled_avg_price"]) if o["filled_avg_price"] else None,
            submitted_at=o["submitted_at"],
        )
        for o in data
    ]


def get_portfolio_history(period: str = "1M", timeframe: str = "1D") -> PortfolioHistory:
    """Real account equity over time, straight from Alpaca -- the actual
    trajectory of the paper account's value, not a backtested curve.
    `period`/`timeframe` follow Alpaca's own conventions (e.g. period:
    "1D", "1W", "1M", "1A"; timeframe: "1Min", "15Min", "1H", "1D")."""
    data = _request("GET", "/v2/account/portfolio/history", params={"period": period, "timeframe": timeframe})
    timestamps = [datetime.fromtimestamp(t, tz=timezone.utc).isoformat() for t in data["timestamp"]]
    return PortfolioHistory(
        timestamps=timestamps,
        equity=data["equity"],
        profit_loss=data["profit_loss"],
        profit_loss_pct=data["profit_loss_pct"],
        base_value=data["base_value"],
    )
