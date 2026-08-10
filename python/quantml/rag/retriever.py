"""Retrieval over a small corpus of sample financial documents -- two
techniques, same `.query(text, top_k)` interface: `Retriever` (TF-IDF +
cosine similarity, classical IR) and `EmbeddingRetriever` (real text
embeddings + cosine similarity, semantic retrieval -- catches queries
phrased differently from the document's own wording, which TF-IDF can't).

The corpus in data/sample_docs/ is synthetic, written for this project to
demonstrate the retrieval + signal pipeline end-to-end without needing a
licensed news/filings feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import embed_texts


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


class EmbeddingRetriever:
    """Same interface as `Retriever`, backed by real text embeddings
    instead of TF-IDF. Doesn't cache the corpus embedding at construction
    time on purpose: the query and the whole corpus are embedded together
    in one `embed_texts()` call on every `.query()`, so they're always
    guaranteed to come from the same backend (Ollama or the hashing
    fallback) -- caching the corpus embedding separately would risk a
    dimension mismatch if Ollama's availability changes between
    construction and query time. Re-embedding a 10-document demo corpus
    on every call is cheap; it wouldn't scale to a large corpus, where
    you'd want to embed once and only re-embed on a backend change."""

    def __init__(self, docs: list[Document]):
        self.docs = docs

    def query(self, text: str, top_k: int = 3) -> list[tuple[Document, float]]:
        if not self.docs:
            return []
        vectors, backend = embed_texts([text] + [d.text for d in self.docs])
        self.last_backend = backend
        query_vec, doc_vecs = vectors[:1], vectors[1:]
        scores = cosine_similarity(query_vec, doc_vecs)[0]
        ranked = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
