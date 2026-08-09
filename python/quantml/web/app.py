"""FastAPI web app: a research dashboard over the backtester/RAG/risk/
volatility/ML-signal stack in `cli.py` and `ml/`.

    cd python
    uvicorn quantml.web.app:app --reload --port 8080

The dashboard endpoints reuse cli.py's exact recipe (same seeds, same
strategy construction) so the numbers they return match the README's
documented CLI output -- this is a view onto the same computation, not a
parallel ad-hoc one. Nothing is cached across requests except the sample-doc
corpus (static files on disk, cheap and safe to load once); every backtest,
VaR simulation, and GARCH fit is recomputed from the same seeded synthetic
data on every call, so every number returned is honest and reproducible
rather than served from a stale cache.

The ML-signal endpoints are deliberately split into two kinds for the same
reason cli.py doesn't re-score the model against its own demo data (see
cli.py's comment on this): `/api/ml-signal` reports the model's TRUE
chronological held-out performance recorded at training time (never
recomputed against data that might overlap what it trained on), while
`/api/ml-signal/predict` does a genuinely fresh, leakage-free thing --
live inference on today's real market data, which isn't a backtest metric
at all and so has no leakage question to begin with.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..data import generate_synthetic_ohlcv, load_real_ohlcv
from ..engine import run_backtest
from ..metrics import monte_carlo_var, summarize
from ..ml.features import build_features
from ..ml.registry import ModelNotTrainedError, load_best_model, load_metadata
from ..rag.retriever import Document, load_corpus
from ..rag.signal import build_signal
from ..risk import RiskLimits, apply_risk_limits
from ..strategies import MovingAverageCrossover, SignalOverlayStrategy
from ..volatility import fit_garch, naive_rolling_vol
from ..walk_forward import run_walk_forward, summarize_walk_forward

# web/app.py sits one level deeper than cli.py (quantml/web/app.py vs
# quantml/cli.py), so this needs one more .parent than cli.py's CORPUS_DIR.
CORPUS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sample_docs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TICKER = "ACME"

# --- cli.py's recipe, reused verbatim so every endpoint's numbers line up
# with the README's documented CLI output. ---------------------------------

_docs_cache: Optional[list[Document]] = None


def _load_corpus_cached() -> list[Document]:
    # The corpus is 10 static .txt files -- safe to load once and reuse,
    # unlike the backtests/sims below which are recomputed every request.
    global _docs_cache
    if _docs_cache is None:
        _docs_cache = load_corpus(CORPUS_DIR)
    return _docs_cache


def _prices():
    return generate_synthetic_ohlcv(seed=7)


def _sentiment(docs: list[Document]):
    return build_signal(docs, tickers=[TICKER], backend="lexicon")


def _baseline_result(prices):
    return run_backtest(prices, MovingAverageCrossover())


def _overlay_result(prices, sentiment):
    overlay = SignalOverlayStrategy(base=MovingAverageCrossover(), signal=sentiment, weight=0.4)
    return run_backtest(prices, overlay)


def _iso_dates(index) -> list[str]:
    return [d.date().isoformat() for d in index]


app = FastAPI(title="QuantML Dashboard", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# --- Dashboard endpoints (read-only, recompute-every-call) -----------------


@app.get("/api/dashboard")
def get_dashboard() -> dict:
    prices = _prices()
    docs = _load_corpus_cached()
    sentiment = _sentiment(docs)

    baseline_result = _baseline_result(prices)
    overlay_result = _overlay_result(prices, sentiment)

    return {
        "dates": _iso_dates(baseline_result.equity_curve.index),
        "baseline_equity": baseline_result.equity_curve.tolist(),
        "overlay_equity": overlay_result.equity_curve.tolist(),
        "baseline_summary": summarize(baseline_result.equity_curve, baseline_result.returns),
        "overlay_summary": summarize(overlay_result.equity_curve, overlay_result.returns),
    }


@app.get("/api/walk-forward")
def get_walk_forward(n_folds: int = Query(default=5, ge=1, le=20)) -> dict:
    prices = _prices()
    wf = run_walk_forward(prices, MovingAverageCrossover(), n_folds=n_folds)
    return {
        "fold_sharpe": wf.fold_sharpe,
        "summary": summarize_walk_forward(wf),
    }


@app.get("/api/var")
def get_var(
    horizon_days: int = Query(default=10, ge=1, le=252),
    confidence: float = Query(default=0.95, gt=0.0, lt=1.0),
) -> dict:
    prices = _prices()
    docs = _load_corpus_cached()
    sentiment = _sentiment(docs)
    overlay_result = _overlay_result(prices, sentiment)
    return monte_carlo_var(overlay_result.returns, horizon_days=horizon_days, confidence=confidence)


@app.get("/api/volatility")
def get_volatility() -> dict:
    prices = _prices()
    returns = prices["close"].pct_change().dropna()
    garch = fit_garch(returns)
    naive = naive_rolling_vol(returns, window=20)
    return {
        "dates": _iso_dates(garch.conditional_vol.index),
        "conditional_vol": garch.conditional_vol.tolist(),
        "forecast_vol": garch.forecast_vol,
        "naive_vol": naive,
    }


@app.get("/api/risk-limits")
def get_risk_limits(max_drawdown: float = Query(default=0.10, gt=0.0, lt=1.0)) -> dict:
    prices = _prices()
    baseline_result = _baseline_result(prices)
    asset_returns = prices["close"].pct_change().fillna(0.0)
    limits = RiskLimits(max_position=1.0, max_drawdown=max_drawdown)
    risk_result = apply_risk_limits(baseline_result.positions, asset_returns, limits)

    return {
        "dates": _iso_dates(baseline_result.equity_curve.index),
        "limited_equity": risk_result.equity_curve.tolist(),
        "unlimited_equity": baseline_result.equity_curve.tolist(),
        "breach_type": risk_result.breach_type,
        "breach_date": risk_result.breach_date.date().isoformat() if risk_result.breach_date is not None else None,
    }


# --- ML signal ---------------------------------------------------------


@app.get("/api/ml-signal")
def get_ml_signal() -> dict:
    """The model's TRUE chronological held-out performance, recorded once
    at training time (ml/train.py) -- not recomputed here, since recomputing
    it against this endpoint's own data could accidentally overlap with
    what the model trained on (see the module docstring)."""
    try:
        metadata = load_metadata()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return metadata


@app.get("/api/ml-signal/predict")
def predict_ml_signal(ticker: str = Query(default="AAPL"), period: str = Query(default="1y")) -> dict:
    """Genuine live inference: pulls today's real market data for `ticker`
    and asks the currently-trained model what it thinks RIGHT NOW. This is
    not a backtest metric, so there's no held-out/leakage question here --
    it's just "run the model on the latest real row," same as
    paper_runner.py does before sizing an order."""
    try:
        model = load_best_model()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    try:
        prices = load_real_ohlcv(ticker, period=period)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    features = build_features(prices)
    if features.empty:
        raise HTTPException(status_code=422, detail=f"Not enough history for {ticker!r} to compute features yet.")

    # Pass the FULL feature history, not just the last row: the GRU model
    # needs a window of prior days to form one prediction (see
    # ml/model.py::_make_sequences), and only its last output row
    # corresponds to "today." The sklearn models don't need this but
    # handle a full table just as well, since they predict row-by-row.
    proba_up = float(model.predict_proba(features)[-1])
    return {
        "ticker": ticker,
        "as_of_date": features.index[-1].date().isoformat(),
        "last_close": float(prices["close"].iloc[-1]),
        "predicted_proba_up": proba_up,
        "suggested_position": max(-1.0, min(1.0, 2 * proba_up - 1)),
    }
