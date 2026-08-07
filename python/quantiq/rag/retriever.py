"""TF-IDF retrieval over a small corpus of sample financial documents.

The corpus in data/sample_docs/ is synthetic, written for this project to
demonstrate the retrieval + signal pipeline end-to-end without needing a
licensed news/filings feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    doc_id: str
    date: str
    ticker: str
    text: str


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        # filename convention: <ticker>_<date>_<slug>.txt
        parts = path.stem.split("_", 2)
        ticker, date = parts[0], parts[1]
        docs.append(Document(doc_id=path.stem, date=date, ticker=ticker, text=path.read_text()))
    return docs


class Retriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([d.text for d in docs]) if docs else None

    def query(self, text: str, top_k: int = 3) -> list[tuple[Document, float]]:
        if not self.docs:
            return []
        q_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
