"""Tests for the trader control service. The background trading thread
itself (autonomous.run) is stubbed out here -- it needs real market data
and a trained model, which is exactly what test_autonomous.py already
covers directly. This file is only about the control surface: does the
service start paused, and do /pause and /resume actually flip it."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from quantml import trader_service


@pytest.fixture
def client():
    # `control` is a module-level singleton (mirrors the real service,
    # where one process controls one loop) -- reset it explicitly rather
    # than relying on test execution order to leave it paused.
    trader_service.control.paused = True
    with patch.object(trader_service.autonomous, "run"):  # never touch real data/network/model
        with TestClient(trader_service.app) as c:
            yield c


def test_starts_paused(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json() == {"paused": True}


def test_resume_then_pause_round_trip(client):
    r = client.post("/resume")
    assert r.status_code == 200
    assert r.json() == {"paused": False}
    assert client.get("/status").json() == {"paused": False}

    r = client.post("/pause")
    assert r.status_code == 200
    assert r.json() == {"paused": True}
    assert client.get("/status").json() == {"paused": True}
