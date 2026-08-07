from pathlib import Path

from quantiq.rag.retriever import Document, Retriever, load_corpus
from quantiq.rag.signal import _lexicon_score, build_signal

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_docs"


def test_load_corpus_parses_filenames():
    docs = load_corpus(CORPUS_DIR)
    assert len(docs) >= 5
    for doc in docs:
        assert doc.ticker == "ACME"
        assert len(doc.date) == 10  # YYYY-MM-DD
        assert doc.text.strip()


def test_retriever_returns_most_relevant_doc():
    docs = [
        Document("a", "2023-01-01", "ACME", "record profit and strong growth this quarter"),
        Document("b", "2023-01-02", "ACME", "lawsuit and product recall weigh on results"),
    ]
    retriever = Retriever(docs)
    results = retriever.query("company profit growth", top_k=1)
    assert results[0][0].doc_id == "a"


def test_lexicon_score_sign():
    assert _lexicon_score("record profit and strong growth beat estimates") > 0
    assert _lexicon_score("lawsuit recall weak decline layoffs miss") < 0
    assert _lexicon_score("the company held a routine meeting") == 0.0


def test_build_signal_bounded_and_indexed():
    docs = load_corpus(CORPUS_DIR)
    signal = build_signal(docs, tickers=["ACME"])
    assert (signal >= -1.0).all() and (signal <= 1.0).all()
    assert signal.index.is_monotonic_increasing


def test_build_signal_ignores_other_tickers():
    docs = load_corpus(CORPUS_DIR)
    signal = build_signal(docs, tickers=["NOPE"])
    assert signal.empty
