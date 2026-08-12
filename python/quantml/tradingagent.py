"""Multi-turn trading assistant agent. Same "LLM proposes, code disposes"
architecture TenantIQ's src/agent.ts used: the LLM only ever decides
WHICH read-only action to take and WHAT ticker, never anything with
financial consequences. Placing an actual trade always requires the
dedicated, password-gated "Run trade now" button (POST /api/trade/run)
-- this agent's action set has no trade action in it at all, so there is
no code path from a chat message to a real order, not just a permission
check that could be misconfigured.

    POST /api/agent/chat  {"message": "...", "history": [...]}

Each turn: the LLM (OpenAI-compatible, same client pattern as
rag/llm.py) reads the user's message plus recent prior turns and returns
a small validated JSON object -- one of a fixed set of read-only actions,
a ticker, and a draft reply. The caller (web/app.py) executes that
action deterministically against the exact same functions the
dashboard's own Predict/Explain/status displays call, then returns the
LLM's phrased reply alongside the real computed data -- the data is
fixed before the model's wording is ever shown, same split as
TenantIQ's landlord auto-reply draft.

Falls back to None (the caller shows a fixed template reply) if the LLM
endpoint is unreachable or returns something that fails validation --
same never-hard-fail contract as rag/llm.py::score_with_llm.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Loads python/.env if present, same convention as paper_trading.py --
# harmless no-op if it doesn't exist or the vars are already set.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL always win when set explicitly
# (how the Azure deployment configures this, via terraform). Otherwise, a
# bare OPENROUTER_API_KEY in .env is enough to point this at OpenRouter
# locally too, without having to also set the OpenAI-style trio by hand.
# With neither, this falls back to a local Ollama server.
_openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL") or (
    "https://openrouter.ai/api/v1" if _openrouter_key else "http://localhost:11434/v1"
)
API_KEY = os.environ.get("OPENAI_API_KEY") or _openrouter_key or "ollama"
MODEL = os.environ.get("OPENAI_MODEL") or (
    "meta-llama/llama-3.3-70b-instruct:free" if _openrouter_key else "qwen2.5:3b"
)

Action = Literal["predict", "explain", "status", "none"]

SYSTEM_PROMPT = (
    "You are a trading research assistant for a stock-prediction dashboard. "
    "You can only read information -- you can never place a trade. Given the "
    "conversation, decide ONE action and respond with strict JSON only, ALWAYS "
    "including all three fields: "
    '{"action": "predict"|"explain"|"status"|"none", "ticker": <string or null>, "reply": <string>}. '
    'Use "predict" for the model\'s live prediction for a ticker, "explain" for feature '
    'importance, "status" for the model\'s recorded held-out performance, or "none" if the '
    "message doesn't need any of those (reply conversationally instead). Default ticker to "
    "AAPL if the user doesn't name one and an action needs one. If asked to place a trade, "
    'set action "none" and explain in the reply that trades go through the dashboard\'s '
    '"Run trade now" button, not chat.\n\n'
    'Example -- user asks "what do you think about microsoft stock?" -> '
    '{"action": "predict", "ticker": "MSFT", "reply": "Let me check the model\'s current read on MSFT."}'
)


class AgentTurn(BaseModel):
    # `reply` deliberately has a default: small local models frequently
    # omit fields even under JSON mode (confirmed empirically against
    # qwen2.5:3b), and a missing reply shouldn't fail the whole turn --
    # the caller substitutes a deterministic template when it's blank,
    # same "guardrails against weak-model mistakes" philosophy TenantIQ
    # used (e.g. treating a bad move-in-date parse as a soft signal, not
    # a hard failure) rather than requiring the model to be perfect.
    action: Action = "none"
    ticker: Optional[str] = None
    reply: str = ""


def interpret_turn(message: str, history: list[dict], timeout: float = 8.0) -> Optional[AgentTurn]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])  # bounded context -- a demo, not a persisted chat history store
    messages.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return AgentTurn.model_validate(json.loads(content))
    except (requests.RequestException, KeyError, json.JSONDecodeError, ValidationError):
        return None
