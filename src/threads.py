"""Reconstruct conversation threads from the flat twcs tweet table.

Each tweet row carries:
  - in_response_to_tweet_id : the parent tweet (NaN if it starts a thread)
  - response_tweet_id       : comma-separated ids of direct replies

A *thread* is the chain from a customer's opening tweet through the alternating
customer/brand replies. We root threads at inbound tweets with no parent and walk
the reply graph depth-first, keeping the earliest-created reply at each step so a
thread is a single linear transcript (branches are rare and we drop the extras).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import RAW_CSV


@dataclass
class Thread:
    thread_id: int                       # tweet_id of the opening customer tweet
    tweet_ids: list[int]
    authors: list[str]
    inbound: list[bool]
    texts: list[str]
    created_at: list[pd.Timestamp]
    brands: set[str] = field(default_factory=set)

    @property
    def brand(self) -> str | None:
        return next(iter(self.brands)) if len(self.brands) == 1 else None

    @property
    def n_turns(self) -> int:
        return len(self.tweet_ids)

    @property
    def customer_opening(self) -> str:
        return self.texts[0]

    @property
    def first_brand_reply(self) -> str | None:
        for a_in, txt in zip(self.inbound[1:], self.texts[1:]):
            if not a_in:
                return txt
        return None

    def transcript(self) -> str:
        lines = []
        for a_in, txt in zip(self.inbound, self.texts):
            who = "CUSTOMER" if a_in else "BRAND"
            lines.append(f"{who}: {txt}")
        return "\n".join(lines)


def load_raw(path=RAW_CSV, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, dtype={"author_id": str})
    df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y")
    return df


def build_threads(df: pd.DataFrame) -> list[Thread]:
    by_id: dict[int, dict] = {
        r.tweet_id: {
            "author": r.author_id,
            "inbound": bool(r.inbound),
            "text": r.text,
            "created": r.created_at,
            "children": [] if pd.isna(r.response_tweet_id)
            else [int(x) for x in str(r.response_tweet_id).split(",") if x.strip().isdigit()],
        }
        for r in df.itertuples(index=False)
    }

    roots = [
        r.tweet_id
        for r in df.itertuples(index=False)
        if bool(r.inbound) and pd.isna(r.in_response_to_tweet_id)
    ]

    threads: list[Thread] = []
    for root in roots:
        chain: list[int] = []
        cur: int | None = root
        seen: set[int] = set()
        while cur is not None and cur in by_id and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            kids = [k for k in by_id[cur]["children"] if k in by_id]
            cur = min(kids, key=lambda k: by_id[k]["created"]) if kids else None

        if len(chain) < 2:
            continue
        nodes = [by_id[c] for c in chain]
        threads.append(
            Thread(
                thread_id=root,
                tweet_ids=chain,
                authors=[n["author"] for n in nodes],
                inbound=[n["inbound"] for n in nodes],
                texts=[n["text"] for n in nodes],
                created_at=[n["created"] for n in nodes],
                brands={n["author"] for n in nodes if not n["inbound"]},
            )
        )
    return threads
