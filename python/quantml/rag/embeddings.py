"""Text embeddings for semantic retrieval -- same pattern TenantIQ used
for listing matching (src/embeddings.ts): try a local Ollama embedding
model first, and fall back to a deterministic feature-hashing embedding
(the hashing trick) if Ollama isn't reachable. No external calls, same
vector for the same text every run, so retrieval degrades gracefully
instead of breaking when no embedding backend is up.

Every text embedded for one retrieval call goes through the SAME path --
the query and the whole corpus are embedded together in one
`embed_texts()` call (see `EmbeddingRetriever.query`), never separately.
Ollama vectors (768-dim) and hashing-trick vectors (256-dim) are never
mixed within a comparison, since their dimensions and spaces aren't
comparable -- if Ollama answers for some texts in a batch but then fails
partway through, that's a real error, not something to silently paper
over with a fallback for just the remaining texts.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"
HASH_DIMENSIONS = 256


def _ollama_embed(text: str, timeout: float = 5.0) -> Optional[np.ndarray]:
    try:
        resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text}, timeout=timeout)
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=float)
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return None


def _hash_embed(text: str, dimensions: int = HASH_DIMENSIONS) -> np.ndarray:
    """The hashing trick: hash each token into a fixed-size vector. Not a
    real semantic embedding, but a stable, deterministic stand-in that
    keeps cosine similarity meaningful when Ollama isn't up -- the same
    text always produces the same vector, with no external dependency."""
    vec = np.zeros(dimensions)
    for token in text.lower().split():
        idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimensions
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_texts(texts: list[str]) -> tuple[np.ndarray, str]:
    """Embeds every text in `texts` through the same backend. Tries
    Ollama for the first text; if that fails, every text in this call
    uses the hashing-trick fallback instead. Returns (matrix,
    backend_name) where backend_name is "ollama" or "hashing_fallback"."""
    if not texts:
        return np.zeros((0, HASH_DIMENSIONS)), "hashing_fallback"

    first = _ollama_embed(texts[0])
    if first is None:
        return np.stack([_hash_embed(t) for t in texts]), "hashing_fallback"

    vectors = [first]
    for t in texts[1:]:
        v = _ollama_embed(t)
        if v is None:
            raise RuntimeError(
                "Ollama answered for an earlier text in this batch but failed on a later one -- "
                "refusing to silently fall back mid-batch, since that would mix incomparable vector spaces."
            )
        vectors.append(v)
    return np.stack(vectors), "ollama"
