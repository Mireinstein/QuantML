"""Turns retrieved documents into a per-date sentiment signal for a ticker.

Two backends:
  - lexicon: deterministic keyword-based scoring. Zero dependencies, always
    available, fully reproducible.
  - llm: optional. Sends each document to an OpenAI-compatible chat endpoint
    (see llm.py) for a structured score, validated with pydantic. Falls back
    to the lexicon backend per-document if the endpoint is unreachable.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .llm import score_with_llm
from .retriever import Document

POSITIVE_WORDS = {
    "beat", "beats", "record", "growth", "upgrade", "outperform", "strong",
    "raised", "raises", "surge", "surged", "profit", "expansion",
}
NEGATIVE_WORDS = {
    "miss", "missed", "downgrade", "underperform", "weak", "cut", "cuts",
    "decline", "declined", "lawsuit", "recall", "loss", "layoffs",
}


def _lexicon_score(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if w.strip(".,!") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!") in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def build_signal(
    docs: list[Document],
    tickers: list[str],
    use_llm: bool = False,
) -> pd.Series:
    """Returns a daily sentiment signal in [-1, 1], indexed by date (averaged
    across a ticker's docs on days with coverage)."""
    by_date: dict[str, list[float]] = defaultdict(list)
    for doc in docs:
        if doc.ticker not in tickers:
            continue
        score = None
        if use_llm:
            score = score_with_llm(doc.text)
        if score is None:
            score = _lexicon_score(doc.text)
        by_date[doc.date].append(score)

    if not by_date:
        return pd.Series(dtype=float)

    dates = sorted(by_date)
    values = [sum(v) / len(v) for v in (by_date[d] for d in dates)]
    idx = pd.to_datetime(dates)
    return pd.Series(values, index=idx).sort_index()
