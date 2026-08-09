"""Optional LLM scoring backend, OpenAI-compatible (defaults to a local,
free Ollama endpoint -- same pattern as the TenantIQ project). Returns None
(triggering a fallback to the lexicon scorer) if no endpoint is
configured/reachable, so the pipeline never hard-fails on a missing key."""
from __future__ import annotations

import json
import os

import requests
from pydantic import BaseModel, Field, ValidationError

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
MODEL = os.environ.get("OPENAI_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = (
    "You are a financial sentiment scorer. Given a short news/filing excerpt, "
    'respond with strict JSON: {"score": <float -1..1>, "rationale": <string>}. '
    "score is -1 for very negative for the company, +1 for very positive, 0 for neutral."
)


class SentimentResponse(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    rationale: str


def score_with_llm(text: str, timeout: float = 5.0) -> float | None:
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = SentimentResponse.model_validate(json.loads(content))
        return parsed.score
    except (requests.RequestException, KeyError, json.JSONDecodeError, ValidationError):
        return None
