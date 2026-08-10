import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from quantml.tradingagent import AgentTurn, interpret_turn


class _FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def raise_for_status(self):
        if not self.ok:
            raise Exception(f"status {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _chat_completion(payload: dict) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload))


def test_interpret_turn_parses_a_valid_response():
    payload = {"action": "predict", "ticker": "MSFT", "reply": "Checking MSFT for you."}
    with patch("quantml.tradingagent.requests.post", return_value=_chat_completion(payload)):
        turn = interpret_turn("what about microsoft?", [])
    assert turn == AgentTurn(action="predict", ticker="MSFT", reply="Checking MSFT for you.")


def test_interpret_turn_defaults_missing_reply_to_empty_string():
    """Real small models frequently omit fields even in JSON mode --
    a missing `reply` must not fail the whole turn."""
    payload = {"action": "predict", "ticker": "AAPL"}
    with patch("quantml.tradingagent.requests.post", return_value=_chat_completion(payload)):
        turn = interpret_turn("what about apple?", [])
    assert turn.action == "predict"
    assert turn.reply == ""


def test_interpret_turn_returns_none_when_endpoint_unreachable():
    import requests

    with patch("quantml.tradingagent.requests.post", side_effect=requests.RequestException("connection refused")):
        turn = interpret_turn("hello", [])
    assert turn is None


def test_interpret_turn_returns_none_on_malformed_json():
    with patch("quantml.tradingagent.requests.post", return_value=_FakeResponse("not json at all")):
        turn = interpret_turn("hello", [])
    assert turn is None


def test_interpret_turn_returns_none_when_action_is_invalid():
    payload = {"action": "place_trade", "ticker": "AAPL", "reply": "done!"}
    with patch("quantml.tradingagent.requests.post", return_value=_chat_completion(payload)):
        turn = interpret_turn("buy some apple stock", [])
    # "place_trade" isn't in the Action Literal -- must fail validation,
    # not silently coerce into something that could be misread as a trade action.
    assert turn is None


def test_agent_turn_action_space_has_no_trade_action():
    """The whole safety property this module relies on: there is no
    action in the schema that could ever be interpreted as "place a
    trade." Assert this directly so a future edit can't quietly add one."""
    import typing

    allowed_actions = typing.get_args(AgentTurn.model_fields["action"].annotation)
    assert allowed_actions == ("predict", "explain", "status", "none")


def test_agent_turn_defaults_to_none_action_and_empty_reply():
    turn = AgentTurn()
    assert turn.action == "none"
    assert turn.reply == ""
    assert turn.ticker is None
