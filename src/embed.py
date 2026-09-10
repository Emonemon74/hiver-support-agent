"""Local sentence-transformers embeddings + a FAISS index helper. No API key."""
from __future__ import annotations

import numpy as np

from src.config import EMBED_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def encode(texts: list[str], *, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    m = _get_model()
    return np.asarray(
        m.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ),
        dtype="float32",
    )


def build_index(vectors: np.ndarray):
    import faiss

    idx = faiss.IndexFlatIP(vectors.shape[1])  # cosine, vectors are normalised
    idx.add(vectors)
    return idx
