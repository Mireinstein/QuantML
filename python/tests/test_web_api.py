"""Tests for the FastAPI dashboard (quantiq.web.app), using FastAPI's
TestClient (same pattern as TenantIQ's ml/tests/test_serve.py)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quantiq.web.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# --- Dashboard endpoints: shape + sanity -----------------------------------


def test_dashboard_shape(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"dates", "baseline_equity", "overlay_equity", "baseline_summary", "overlay_summary"}
    assert len(body["dates"]) > 0
    assert len(body["dates"]) == len(body["baseline_equity"]) == len(body["overlay_equity"])
    for summary in (body["baseline_summary"], body["overlay_summary"]):
        assert set(summary.keys()) == {"sharpe", "cagr", "max_drawdown", "win_rate", "final_equity"}
        assert isinstance(summary["sharpe"], float)


def test_walk_forward_shape(client):
    r = client.get("/api/walk-forward?n_folds=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["fold_sharpe"]) == 5
    assert body["summary"]["n_folds"] == 5
    assert set(body["summary"].keys()) == {"n_folds", "mean_sharpe", "std_sharpe", "worst_fold_sharpe", "worst_fold_drawdown"}


def test_var_shape(client):
    r = client.get("/api/var?horizon_days=10&confidence=0.95")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"horizon_days", "confidence", "var", "cvar", "n_sims"}
    assert body["var"] >= 0  # VaR reported as a positive loss magnitude
    assert body["cvar"] >= body["var"]  # CVaR is the tail average, at least as bad as VaR


def test_volatility_shape(client):
    r = client.get("/api/volatility")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"dates", "conditional_vol", "forecast_vol", "naive_vol"}
    assert len(body["dates"]) == len(body["conditional_vol"])
    assert len(body["dates"]) > 0
    assert body["forecast_vol"] > 0
    assert body["naive_vol"] > 0


def test_risk_limits_breaches_with_tight_drawdown(client):
    r = client.get("/api/risk-limits?max_drawdown=0.02")
    assert r.status_code == 200
    body = r.json()
    assert body["breach_type"] == "max_drawdown"
    assert body["breach_date"] is not None


def test_risk_limits_no_breach_with_loose_drawdown(client):
    r = client.get("/api/risk-limits?max_drawdown=0.99")
    assert r.status_code == 200
    body = r.json()
    assert body["breach_type"] is None
    assert body["breach_date"] is None


def test_risk_limits_shape(client):
    r = client.get("/api/risk-limits?max_drawdown=0.10")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"dates", "limited_equity", "unlimited_equity", "breach_type", "breach_date"}
    assert len(body["dates"]) == len(body["limited_equity"]) == len(body["unlimited_equity"])


# --- ML signal ---------------------------------------------------------


def test_ml_signal_returns_503_when_no_model_trained(client, monkeypatch, tmp_path):
    import quantiq.web.app as app_module

    monkeypatch.setattr(app_module, "load_metadata", _raise_not_trained)
    r = client.get("/api/ml-signal")
    assert r.status_code == 503


def test_ml_signal_shape_when_a_model_exists(client):
    r = client.get("/api/ml-signal")
    # Either a real trained model exists in this checkout (200, real shape)
    # or none has been trained yet in this environment (503, handled above) --
    # both are valid states for a fresh clone; only assert the shape when one exists.
    if r.status_code == 200:
        body = r.json()
        assert {"version", "model_type", "held_out_auc", "held_out_backtest", "data_source"} <= set(body.keys())


def test_ml_signal_predict_rejects_a_ticker_with_no_history(client):
    r = client.get("/api/ml-signal/predict?ticker=THIS-IS-NOT-A-REAL-TICKER-XYZ")
    # Either the model isn't trained yet (503) or yfinance correctly finds
    # nothing for a nonsense ticker (422) -- both are the correct failure
    # mode for this input, never a 200.
    assert r.status_code in (422, 503)


def _raise_not_trained(*args, **kwargs):
    from quantiq.ml.registry import ModelNotTrainedError

    raise ModelNotTrainedError("no model for this test")


# --- Static frontend -----------------------------------------------------


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<title>QuantIQ Dashboard</title>" in r.text


def test_static_js_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "loadMlSignal" in r.text
