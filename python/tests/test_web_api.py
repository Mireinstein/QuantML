"""Tests for the FastAPI dashboard (quantml.web.app), using FastAPI's
TestClient (same pattern as TenantIQ's ml/tests/test_serve.py)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quantml.web.app import app


@pytest.fixture(autouse=True)
def clean_dashboard_auth_env(monkeypatch):
    # paper_trading.py loads python/.env at import time, which -- once a
    # real developer has DASHBOARD_USERNAME/DASHBOARD_PASSWORD set there
    # for an actual deployment -- would otherwise leak into every test's
    # os.environ and turn auth on everywhere. Force it off by default;
    # individual tests opt back in with monkeypatch.setenv.
    monkeypatch.delenv("DASHBOARD_USERNAME", raising=False)
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)


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
    import quantml.web.app as app_module

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
    from quantml.ml.registry import ModelNotTrainedError

    raise ModelNotTrainedError("no model for this test")


# --- Explainability ---------------------------------------------------


def test_explain_returns_503_when_no_model_trained(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "load_best_model", _raise_not_trained)
    r = client.get("/api/ml-signal/explain?ticker=AAPL")
    assert r.status_code == 503


def test_explain_rejects_a_ticker_with_no_history(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "load_best_model", lambda: object())
    r = client.get("/api/ml-signal/explain?ticker=THIS-IS-NOT-A-REAL-TICKER-XYZ")
    assert r.status_code in (422, 503)


# --- Auth on the action endpoints only -------------------------------------


def test_trade_run_open_when_dashboard_password_unset(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.delenv("DASHBOARD_USERNAME", raising=False)
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr(app_module, "load_best_model", _raise_not_trained)
    r = client.post("/api/trade/run?ticker=AAPL")
    assert r.status_code == 503  # reached the real handler, not blocked at 401


def test_trade_run_requires_auth_when_configured(client, monkeypatch):
    monkeypatch.setenv("DASHBOARD_USERNAME", "admire")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret123")
    r = client.post("/api/trade/run?ticker=AAPL")
    assert r.status_code == 401


def test_trade_run_rejects_wrong_credentials(client, monkeypatch):
    monkeypatch.setenv("DASHBOARD_USERNAME", "admire")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret123")
    r = client.post("/api/trade/run?ticker=AAPL", auth=("admire", "wrong-password"))
    assert r.status_code == 401


def test_trade_run_accepts_correct_credentials(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setenv("DASHBOARD_USERNAME", "admire")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret123")
    monkeypatch.setattr(app_module, "load_best_model", _raise_not_trained)
    r = client.post("/api/trade/run?ticker=AAPL", auth=("admire", "secret123"))
    assert r.status_code == 503  # past auth, reached the real (stubbed) handler


def test_read_only_endpoints_stay_open_even_when_auth_is_configured(client, monkeypatch):
    """The whole point: performance/metrics/history must never require a
    login, only the endpoints that actually place a trade or start/stop
    the bot."""
    monkeypatch.setenv("DASHBOARD_USERNAME", "admire")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret123")
    for path in ("/api/dashboard", "/api/autonomous/activity", "/api/autonomous/status", "/"):
        r = client.get(path)
        assert r.status_code != 401, f"{path} must not require auth"


# --- Autonomous start/stop --------------------------------------------


def test_autonomous_status_reports_unconfigured_without_trader_url(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "TRADER_INTERNAL_URL", None)
    r = client.get("/api/autonomous/status")
    assert r.status_code == 200
    assert r.json() == {"configured": False, "running": None}


def test_autonomous_start_returns_503_without_trader_url(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "TRADER_INTERNAL_URL", None)
    r = client.post("/api/autonomous/start")
    assert r.status_code == 503


def test_autonomous_start_requires_auth_when_configured(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "TRADER_INTERNAL_URL", "http://trader.internal")
    monkeypatch.setenv("DASHBOARD_USERNAME", "admire")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret123")
    r = client.post("/api/autonomous/start")
    assert r.status_code == 401


def test_autonomous_start_proxies_to_trader_service(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "TRADER_INTERNAL_URL", "http://trader.internal")

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"paused": False}

    calls = []

    def _fake_post(url, timeout):  # noqa: ARG001
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(app_module.http, "post", _fake_post)
    r = client.post("/api/autonomous/start")
    assert r.status_code == 200
    assert r.json() == {"running": True}
    assert calls == ["http://trader.internal/resume"]


def test_autonomous_stop_proxies_to_trader_service(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "TRADER_INTERNAL_URL", "http://trader.internal")

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"paused": True}

    calls = []

    def _fake_post(url, timeout):  # noqa: ARG001
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(app_module.http, "post", _fake_post)
    r = client.post("/api/autonomous/stop")
    assert r.status_code == 200
    assert r.json() == {"running": False}
    assert calls == ["http://trader.internal/pause"]


# --- On-demand trading + autonomous activity ------------------------------


def test_run_trade_returns_503_when_no_model_trained(client, monkeypatch):
    import quantml.web.app as app_module

    monkeypatch.setattr(app_module, "load_best_model", _raise_not_trained)
    r = client.post("/api/trade/run?ticker=AAPL")
    assert r.status_code == 503


def test_run_trade_returns_502_on_paper_trading_error(client, monkeypatch):
    import quantml.web.app as app_module
    from quantml.paper_trading import PaperTradingError

    monkeypatch.setattr(app_module, "load_best_model", lambda: object())

    def _raise_paper_error(*args, **kwargs):
        raise PaperTradingError("no credentials configured")

    monkeypatch.setattr(app_module, "rebalance", _raise_paper_error)
    r = client.post("/api/trade/run?ticker=AAPL")
    assert r.status_code == 502


def test_run_trade_serializes_order_result(client, monkeypatch):
    import quantml.web.app as app_module
    from quantml.paper_trading import OrderResult

    monkeypatch.setattr(app_module, "load_best_model", lambda: object())
    monkeypatch.setattr(
        app_module,
        "rebalance",
        lambda *a, **k: {
            "ticker": "AAPL",
            "last_close": 200.0,
            "current_shares": 0,
            "target_shares": 5,
            "delta": 5,
            "order": OrderResult(id="1", symbol="AAPL", qty=5, side="buy", status="accepted"),
        },
    )
    r = client.post("/api/trade/run?ticker=AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["order"] == {"id": "1", "symbol": "AAPL", "qty": 5, "side": "buy", "status": "accepted"}


def test_autonomous_activity_empty_when_never_run(client, monkeypatch, tmp_path):
    from quantml import autonomous

    monkeypatch.setattr(autonomous, "LOG_PATH", tmp_path / "does_not_exist.jsonl")
    r = client.get("/api/autonomous/activity")
    assert r.status_code == 200
    assert r.json() == {"activity": []}


def test_autonomous_trades_returns_503_without_credentials(client, monkeypatch):
    import quantml.web.app as app_module
    from quantml.paper_trading import PaperTradingError

    def _raise(*args, **kwargs):
        raise PaperTradingError("missing credentials")

    monkeypatch.setattr(app_module, "list_orders", _raise)
    r = client.get("/api/autonomous/trades")
    assert r.status_code == 503


def test_autonomous_trades_shape(client, monkeypatch):
    import quantml.web.app as app_module
    from quantml.paper_trading import OrderRecord

    monkeypatch.setattr(
        app_module,
        "list_orders",
        lambda **kwargs: [
            OrderRecord(
                id="o1", symbol="AAPL", side="buy", qty=5, status="filled",
                filled_qty=5, filled_avg_price=201.5, submitted_at="2026-08-09T15:08:37Z",
            )
        ],
    )
    r = client.get("/api/autonomous/trades")
    assert r.status_code == 200
    body = r.json()
    assert body["trades"][0]["filled_avg_price"] == 201.5


def test_autonomous_equity_returns_503_without_credentials(client, monkeypatch):
    import quantml.web.app as app_module
    from quantml.paper_trading import PaperTradingError

    def _raise(*args, **kwargs):
        raise PaperTradingError("missing credentials")

    monkeypatch.setattr(app_module, "get_portfolio_history", _raise)
    r = client.get("/api/autonomous/equity")
    assert r.status_code == 503


def test_autonomous_equity_shape(client, monkeypatch):
    import quantml.web.app as app_module
    from quantml.paper_trading import PortfolioHistory

    monkeypatch.setattr(
        app_module,
        "get_portfolio_history",
        lambda **kwargs: PortfolioHistory(
            timestamps=["2026-08-01T00:00:00+00:00"], equity=[100000.0], profit_loss=[0.0],
            profit_loss_pct=[0.0], base_value=100000.0,
        ),
    )
    r = client.get("/api/autonomous/equity")
    assert r.status_code == 200
    assert r.json()["equity"] == [100000.0]


def test_autonomous_generations_filters_to_promotion_events(client, monkeypatch):
    from quantml import autonomous

    monkeypatch.setattr(
        autonomous,
        "recent_activity",
        lambda n: [
            {"event": "cycle", "cycle": 1},
            {"event": "model_promoted", "generation": 1, "auc": 0.56},
            {"event": "retrain_rejected", "reasons": ["Sharpe regressed"]},
        ],
    )
    r = client.get("/api/autonomous/generations")
    assert r.status_code == 200
    events = [g["event"] for g in r.json()["generations"]]
    assert events == ["model_promoted", "retrain_rejected"]


# --- Static frontend -----------------------------------------------------


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<title>QuantML Dashboard</title>" in r.text


def test_static_js_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "loadMlSignal" in r.text
