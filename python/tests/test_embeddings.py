from unittest.mock import patch

import numpy as np
import pytest

from quantml.rag.embeddings import _hash_embed, embed_texts
from quantml.rag.retriever import Document, EmbeddingRetriever


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def raise_for_status(self):
        if not self.ok:
            raise Exception(f"status {self.status_code}")

    def json(self):
        return self._json


def test_hash_embed_is_deterministic_and_unit_normalized():
    v1 = _hash_embed("record profit and strong growth")
    v2 = _hash_embed("record profit and strong growth")
    assert np.array_equal(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9


def test_hash_embed_differs_for_different_text():
    v1 = _hash_embed("record profit and strong growth")
    v2 = _hash_embed("lawsuit and product recall")
    assert not np.array_equal(v1, v2)


def test_embed_texts_falls_back_to_hashing_when_ollama_unreachable():
    import requests

    with patch.object(requests, "post", side_effect=requests.RequestException("connection refused")):
        matrix, backend = embed_texts(["hello", "world"])
    assert backend == "hashing_fallback"
    assert matrix.shape == (2, 256)


def test_embed_texts_uses_ollama_when_reachable():
    import requests

    fake = _FakeResponse({"embedding": [0.1, 0.2, 0.3]})
    with patch.object(requests, "post", return_value=fake):
        matrix, backend = embed_texts(["hello", "world"])
    assert backend == "ollama"
    assert matrix.shape == (2, 3)


def test_embed_texts_raises_rather_than_mix_backends_mid_batch():
    """If Ollama answers the first text but then fails on a later one in
    the same batch, that must be a loud error, not a silent fallback for
    just the remaining texts (which would mix incomparable vector spaces
    -- 768-dim Ollama vectors with 256-dim hashing vectors)."""
    import requests

    responses = [_FakeResponse({"embedding": [0.1, 0.2]}), requests.RequestException("down")]

    def _side_effect(*args, **kwargs):  # noqa: ARG001
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with patch.object(requests, "post", side_effect=_side_effect):
        with pytest.raises(RuntimeError, match="mid-batch"):
            embed_texts(["first", "second"])


def test_embed_texts_empty_input():
    matrix, backend = embed_texts([])
    assert matrix.shape == (0, 256)
    assert backend == "hashing_fallback"


def test_embedding_retriever_ranks_most_relevant_doc_first():
    import requests

    docs = [
        Document("a", "2023-01-01", "ACME", "record profit and strong growth this quarter"),
        Document("b", "2023-01-02", "ACME", "lawsuit and product recall weigh on results"),
    ]
    with patch.object(requests, "post", side_effect=requests.RequestException("no ollama in test env")):
        retriever = EmbeddingRetriever(docs)
        results = retriever.query("company profit growth", top_k=1)
    assert results[0][0].doc_id == "a"
    assert retriever.last_backend == "hashing_fallback"


def test_embedding_retriever_returns_empty_for_no_docs():
    retriever = EmbeddingRetriever([])
    assert retriever.query("anything") == []
