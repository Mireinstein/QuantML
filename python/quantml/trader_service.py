"""Control service for the autonomous trading loop, deployed as its own
Azure Container App with INTERNAL-ONLY ingress -- never reachable from
the public internet, only from other apps in the same Container Apps
environment (i.e. the dashboard). Runs `autonomous.run()` in a background
thread and exposes /pause, /resume, /status so the (password-protected)
dashboard can start and stop real trading without needing direct network
access to this container.

    uvicorn quantml.trader_service:app --host 0.0.0.0 --port 8080

Starts PAUSED: the loop thread is running (so its startup work -- loading
data, the model -- happens once at container boot, not on every
start/resume), but it does nothing and places no orders until something
calls POST /resume. A freshly-deployed or freshly-restarted container
should never trade without an explicit "start."
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import autonomous

control = autonomous.RunControl(paused=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ticker = os.environ.get("TRADER_TICKER", "AAPL")
    check_interval_seconds = int(os.environ.get("TRADER_CHECK_INTERVAL_SECONDS", str(autonomous.DEFAULT_CHECK_INTERVAL_SECONDS)))

    thread = threading.Thread(
        target=autonomous.run,
        kwargs={
            "ticker": ticker,
            "check_interval_seconds": check_interval_seconds,
            "control": control,
        },
        daemon=True,
    )
    thread.start()
    yield


app = FastAPI(title="QuantML Trader Control", version="1.0", lifespan=lifespan)


@app.get("/status")
def status() -> dict:
    return {"paused": control.paused}


@app.post("/pause")
def pause() -> dict:
    control.paused = True
    return {"paused": True}


@app.post("/resume")
def resume() -> dict:
    control.paused = False
    return {"paused": False}
