"""Phase 4a: draw a diverse, stratified sample from the held-out threads to be
hand-labelled into the golden set.

Strategy (no gold labels exist yet, so we stratify on cheap signals):
  * embed openings locally, KMeans(k=10) for topical diversity
  * cross with structural strata: question vs statement, has link, length bucket,
    and whether the brand's actual reply was a "DM us" hand-off
  * sample ~TARGET with a per-cluster floor so rare topics are not missed
  * drop near-duplicates (cosine > 0.95) so labels aren't wasted on repeats

Run: python -m src.golden.sample
Output: data/golden_sample.parquet  (the rows to label)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import DATA, SEED
from src.embed import encode

TARGET = 200
K = 10
PER_CLUSTER_FLOOR = 12


def _structural_stratum(row) -> str:
    q = "q" if "?" in row.customer_opening else "s"
    u = "u" if "<URL>" in row.customer_opening else "n"
    n = len(row.customer_opening)
    lb = "short" if n < 80 else "long" if n > 180 else "mid"
    h = "handoff" if row.reply_is_handoff else "resolved"
    return f"{q}-{u}-{lb}-{h}"


def main() -> None:
    hold = pd.read_parquet(DATA / "holdout.parquet").reset_index(drop=True)
    vecs = encode(hold.customer_opening.tolist(), show_progress=True)

    km = KMeans(K, random_state=SEED, n_init=10).fit(vecs)
    hold["cluster"] = km.labels_
    hold["stratum"] = [_structural_stratum(r) for r in hold.itertuples(index=False)]

    picked: list[int] = []
    # proportional allocation across clusters, with a floor
    sizes = hold.cluster.value_counts().sort_index()
    alloc = np.maximum(
        PER_CLUSTER_FLOOR, np.round(TARGET * sizes / sizes.sum()).astype(int)
    )
    for c, n_take in alloc.items():
        pool = hold[hold.cluster == c]
        # within a cluster, spread across structural strata
        order = (
            pool.groupby("stratum", group_keys=False)
            .apply(lambda g: g.sample(frac=1, random_state=SEED))
            .index.tolist()
        )
        picked.extend(order[: min(n_take, len(order))])

    sample = hold.loc[picked].drop_duplicates("thread_id")

    # near-duplicate removal
    sv = encode(sample.customer_opening.tolist())
    keep, seen = [], np.zeros((0, sv.shape[1]), dtype="float32")
    for i in range(len(sv)):
        if seen.shape[0] and float((seen @ sv[i]).max()) > 0.95:
            continue
        keep.append(i)
        seen = np.vstack([seen, sv[i]])
    sample = sample.iloc[keep]

    if len(sample) > TARGET:
        sample = sample.sample(TARGET, random_state=SEED)
    sample = sample.sort_values("created_at").reset_index(drop=True)

    cols = ["thread_id", "created_at", "customer_opening", "first_brand_reply",
            "transcript", "reply_is_handoff", "cluster", "stratum"]
    sample[cols].to_parquet(DATA / "golden_sample.parquet")
    print(f"sampled {len(sample)} threads -> data/golden_sample.parquet")
    print(sample.cluster.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
