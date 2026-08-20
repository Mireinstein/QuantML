from unittest.mock import MagicMock, patch

from quantml.rag.news import load_ticker_news
from quantml.rag.signal import build_signal


def _fake_ticker(news_items):
    ticker = MagicMock()
    ticker.news = news_items
    return ticker


def test_load_ticker_news_maps_yahoo_items_to_documents():
    items = [
        {
            "id": "abc",
            "content": {
                "id": "abc",
                "title": "AAPL beats estimates on strong iPhone demand",
                "summary": "Record quarterly profit and raised guidance.",
                "pubDate": "2026-08-18T14:40:36Z",
            },
        }
    ]
    with patch("quantml.rag.news.yf.Ticker", return_value=_fake_ticker(items)):
        docs = load_ticker_news("aapl")

    assert len(docs) == 1
    assert docs[0].ticker == "AAPL"
    assert docs[0].date == "2026-08-18"
    assert "beats estimates" in docs[0].text
    assert "raised guidance" in docs[0].text


def test_load_ticker_news_skips_malformed_items():
    items = [
        {"id": "1", "content": {"title": "", "pubDate": "2026-08-18T00:00:00Z"}},  # no title
        {"id": "2", "content": {"title": "Valid headline", "pubDate": "bad"}},  # unusable date
        {"id": "3"},  # no content at all
        {"id": "4", "content": {"title": "Kept", "summary": "s", "pubDate": "2026-08-17T09:00:00Z"}},
    ]
    with patch("quantml.rag.news.yf.Ticker", return_value=_fake_ticker(items)):
        docs = load_ticker_news("MSFT")
    assert [d.text.split(".")[0] for d in docs] == ["Kept"]


def test_load_ticker_news_returns_empty_on_network_failure():
    with patch("quantml.rag.news.yf.Ticker", side_effect=ConnectionError("offline")):
        assert load_ticker_news("AAPL") == []


def test_news_documents_flow_through_build_signal():
    """The point of the module: real-ticker news must produce a non-empty
    sentiment series through the exact same build_signal path the sample
    corpus uses -- this is what was broken before (any real ticker got an
    empty series and a permanently neutral overlay)."""
    items = [
        {
            "id": "up",
            "content": {
                "title": "NVDA surges past estimates with record profit growth",
                "summary": "Analysts raised targets citing strong demand.",
                "pubDate": "2026-08-18T10:00:00Z",
            },
        },
        {
            "id": "down",
            "content": {
                "title": "NVDA hit with downgrade as demand looks weak",
                "summary": "Analysts cut estimates citing decline in orders.",
                "pubDate": "2026-08-17T10:00:00Z",
            },
        },
    ]
    with patch("quantml.rag.news.yf.Ticker", return_value=_fake_ticker(items)):
        docs = load_ticker_news("NVDA")

    signal = build_signal(docs, tickers=["NVDA"])
    assert len(signal) == 2
    assert signal.loc["2026-08-18"] > 0  # positive-lexicon day
    assert signal.loc["2026-08-17"] < 0  # negative-lexicon day

    # Ticker matching is case-insensitive -- `--ticker nvda` must produce
    # the same signal, not a silently empty one.
    assert len(build_signal(docs, tickers=["nvda"])) == 2
