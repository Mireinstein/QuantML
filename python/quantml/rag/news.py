"""Real news headlines for real tickers, as retriever Documents.

The bundled sample corpus (data/sample_docs/) only covers the fictional
demo ticker ACME, so a sentiment signal built from it is empty for any
real ticker. This module fills that gap with Yahoo Finance's news feed
for the requested ticker -- real, current headlines, scored by the same
backends as the corpus docs. Coverage is recent-only (Yahoo returns
roughly the last few days of stories), so on a multi-year backtest the
sentiment overlay tilts only the recent tail; that reflects what's
actually knowable, and the daily live-trading path only needs today's
signal anyway.

Returns [] on any failure (no network, delisted ticker, API shape
change) -- same degrade-don't-crash convention as the rest of the
data layer.
"""
from __future__ import annotations

import yfinance as yf

from .retriever import Document


def load_ticker_news(ticker: str, limit: int = 20) -> list[Document]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    docs = []
    for item in items[:limit]:
        content = item.get("content") or {}
        title = content.get("title") or ""
        pub_date = content.get("pubDate") or ""
        if not title or len(pub_date) < 10:
            continue
        summary = content.get("summary") or ""
        date = pub_date[:10]  # ISO "YYYY-MM-DDTHH:MM:SSZ" -> "YYYY-MM-DD"
        docs.append(
            Document(
                doc_id=f"{ticker.upper()}_{date}_news_{content.get('id', len(docs))}",
                date=date,
                ticker=ticker.upper(),
                text=f"{title}. {summary}".strip(),
            )
        )
    return docs
