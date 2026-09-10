"""Retrieval over the historical Delta corpus: given a new customer message, find
the most similar past *resolved* threads and return their (opening, brand reply)
pairs as precedent for drafting.

The index is a FAISS inner-product index over locally-embedded openings; vectors
are cached to data/corpus_vecs.npy so it builds once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DATA
from src.embed import build_index, encode

_VECS = DATA / "corpus_vecs.npy"


@dataclass
class Precedent:
    opening: str
    reply: str
    score: float
    is_handoff: bool


class Retriever:
    def __init__(self) -> None:
        self.corpus = pd.read_parquet(DATA / "corpus.parquet").reset_index(drop=True)
        if _VECS.exists() and len(np.load(_VECS)) == len(self.corpus):
            vecs = np.load(_VECS)
        else:
            vecs = encode(self.corpus.customer_opening.tolist(), show_progress=True)
            np.save(_VECS, vecs)
        self.index = build_index(vecs)

    def search(self, message: str, k: int = 5) -> list[Precedent]:
        q = encode([message])
        scores, idx = self.index.search(q, k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            row = self.corpus.iloc[int(i)]
            out.append(Precedent(
                opening=row.customer_opening,
                reply=row.first_brand_reply,
                score=float(s),
                is_handoff=bool(row.reply_is_handoff),
            ))
        return out


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
