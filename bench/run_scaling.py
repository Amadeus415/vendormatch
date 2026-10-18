"""Runtime of the non-LLM pipeline from 1k to 50k records, stage by stage.

Uses data/scale.csv (bench.generate --scale): companies not in dev, each with more records than in
dev/test so that 50k records exist at all (SEC has ~6.5k operating companies). For each size, take
whole companies in a fixed random order until the record budget is reached, then run every stage.

    uv run python -m bench.run_scaling
"""

from __future__ import annotations

import random
import time
from collections import defaultdict

from vendormatch.blocking import candidates
from vendormatch.cluster import average_link
from vendormatch.features import pair_features, prepare
from vendormatch.metrics import cluster_scores, pair_scores, true_pairs
from vendormatch.pipeline import _center
from vendormatch.scorer import PairScorer

from .common import RESULTS, load_json, load_split, metadata, save_json

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
REPEATS = 3  # other jobs share this machine; report the median run per size


def subset(records: list[dict], n: int) -> list[dict]:
    by_entity = defaultdict(list)
    for r in records:
        by_entity[r["entity_id"]].append(r)
    ids = sorted(by_entity)
    random.Random(0).shuffle(ids)
    out = []
    for e in ids:
        if len(out) >= n:
            break
        out.extend(by_entity[e][: n - len(out)])
    return out


def run(records: list[dict], scorer: PairScorer, k: int) -> dict:
    t = {}
    t0 = time.perf_counter()
    prep = prepare(records)
    t["normalize"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pairs = candidates(prep.variants, k=k)
    t["block"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    X = pair_features(prep, pairs)
    t["features"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = scorer.predict(X)
    t["score"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    centered = _center(s, scorer.threshold)
    labels = average_link(len(records), dict(zip(pairs, centered.tolist(), strict=True)))
    t["cluster"] = time.perf_counter() - t0
    truth_ids = [r["entity_id"] for r in records]
    lo, hi = scorer.band
    pair_f1 = pair_scores([p for p, v in zip(pairs, centered, strict=True) if v >= 0.5], true_pairs(truth_ids))
    return {
        "records": len(records),
        "companies": len(set(truth_ids)),
        "candidates": len(pairs),
        "band_pairs": int(((s >= lo) & (s <= hi)).sum()),
        "seconds": t,
        "total_seconds": sum(t.values()),
        "records_per_second": len(records) / sum(t.values()),
        "pair_f1": pair_f1["f1"],
        "cluster": cluster_scores(labels, truth_ids),
    }


def main() -> None:
    k = load_json(RESULTS / "blocking.json")["chosen_k"]
    scorer = PairScorer.load()
    records = load_split("scale")
    run(subset(records, 500), scorer, k)  # warm up imports and caches
    rows = []
    for n in SIZES:
        part = subset(records, n)
        runs = sorted((run(part, scorer, k) for _ in range(REPEATS)), key=lambda r: r["total_seconds"])
        r = runs[len(runs) // 2]
        r["all_total_seconds"] = [x["total_seconds"] for x in runs]
        rows.append(r)
        stages = ", ".join(f"{s} {v:.1f}s" for s, v in r["seconds"].items())
        print(f"{n:6d} records: {r['total_seconds']:6.1f}s ({r['records_per_second']:,.0f} rec/s) {stages}; "
              f"pair F1 {r['pair_f1']:.3f}")  # fmt: skip
    save_json(RESULTS / "scaling.json", {"meta": metadata(k=k, repeats=REPEATS), "rows": rows})


if __name__ == "__main__":
    main()
