from unittest.mock import patch

import pytest

from quantml import paper_trading as pt


@pytest.fixture(autouse=True)
def alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-secret")


class _FakeResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self._json = json_data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text or "{}"

    def json(self):
        return self._json


def test_paper_base_url_is_hardcoded_to_the_paper_subdomain():
    # The whole safety property of this module rests on this never being
    # configurable -- assert it directly so a future edit can't quietly
    # turn it into an env var.
    assert pt.PAPER_BASE_URL == "https://paper-api.alpaca.markets"


def test_missing_credentials_raise_a_clear_error(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(pt.PaperTradingError, match="ALPACA_API_KEY_ID"):
        pt.get_account()


def test_get_account_parses_a_real_shaped_response():
    fake = _FakeResponse({"cash": "1000.50", "portfolio_value": "5000.00", "equity": "5000.00", "buying_power": "2000.00"})
    with patch.object(pt.requests, "request", return_value=fake) as mock_request:
        account = pt.get_account()
    assert account.cash == 1000.50
    assert account.equity == 5000.00
    called_url = mock_request.call_args.args[1]
    assert called_url.startswith(pt.PAPER_BASE_URL)


def test_get_position_returns_none_on_404_not_an_error():
    fake = _FakeResponse({}, status_code=404, text="position does not exist")
    with patch.object(pt.requests, "request", return_value=fake):
        assert pt.get_position("AAPL") is None


def test_get_position_returns_parsed_position_when_present():
    fake = _FakeResponse(
        {"symbol": "AAPL", "qty": "10", "avg_entry_price": "150.0", "market_value": "1550.0", "unrealized_pl": "50.0"}
    )
    with patch.object(pt.requests, "request", return_value=fake):
        pos = pt.get_position("AAPL")
    assert pos.symbol == "AAPL"
    assert pos.qty == 10


def test_submit_market_order_rejects_non_positive_qty():
    with pytest.raises(pt.PaperTradingError, match="qty must be positive"):
        pt.submit_market_order("AAPL", qty=0, side="buy")


def test_submit_market_order_posts_expected_payload():
    fake = _FakeResponse({"id": "abc123", "symbol": "AAPL", "qty": "5", "side": "buy", "status": "accepted"})
    with patch.object(pt.requests, "request", return_value=fake) as mock_request:
        result = pt.submit_market_order("AAPL", qty=5, side="buy")

    assert result.id == "abc123"
    assert result.status == "accepted"
    _, kwargs = mock_request.call_args
    assert kwargs["json"]["symbol"] == "AAPL"
    assert kwargs["json"]["side"] == "buy"
    assert kwargs["json"]["type"] == "market"


def test_non_ok_response_raises_paper_trading_error():
    fake = _FakeResponse({}, status_code=500, text="internal error")
    with patch.object(pt.requests, "request", return_value=fake):
        with pytest.raises(pt.PaperTradingError, match="500"):
            pt.get_account()
