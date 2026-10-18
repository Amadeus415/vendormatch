"""Stage 2 benchmark: how much pair recall does each blocker keep, and at what candidate count?

    uv run python -m bench.run_blocking
"""

from __future__ import annotations

import time
from collections import Counter

from vendormatch.blocking import acronym_blocking, reduction_ratio, tfidf_knn, token_blocking
from vendormatch.metrics import true_pairs
from vendormatch.normalize import name_variants

from .common import RESULTS, load_split, metadata, save_json

KS = [1, 2, 3, 5, 10, 20, 50]
MAX_DFS = [2, 5, 10, 25, 50, 100, 250]


def evaluate(pairs, truth, n, seconds, **params) -> dict:
    found = len(set(pairs) & truth)
    return {
        **params,
        "candidates": len(pairs),
        "recall": found / len(truth),
        "reduction_ratio": reduction_ratio(n, len(pairs)),
        "seconds": seconds,
    }


def run_split(name: str) -> dict:
    records = load_split(name)
    t0 = time.perf_counter()
    variants = [name_variants(r["name"]) for r in records]
    t_norm = time.perf_counter() - t0
    truth = true_pairs([r["entity_id"] for r in records])
    n = len(records)
    rows = []

    t0 = time.perf_counter()
    acr = acronym_blocking(variants)
    t_acr = time.perf_counter() - t0
    rows.append(evaluate(acr, truth, n, t_acr, method="acronym"))

    for k in KS:
        t0 = time.perf_counter()
        knn = tfidf_knn(variants, k=k)
        dt = time.perf_counter() - t0
        rows.append(evaluate(set(knn), truth, n, dt, method="tfidf_knn", k=k))
        rows.append(evaluate(set(knn) | acr, truth, n, dt + t_acr, method="tfidf_knn+acronym", k=k))
    for max_df in MAX_DFS:
        t0 = time.perf_counter()
        tok = token_blocking(variants, max_df=max_df)
        rows.append(evaluate(tok, truth, n, time.perf_counter() - t0, method="token", max_df=max_df))

    print(f"{name}: {n} records, {len(truth)} true pairs, normalize {t_norm:.2f}s")
    for r in rows:
        print(f"  {r['method']:18} k={r.get('k', '-')!s:3} max_df={r.get('max_df', '-')!s:4} "
              f"cand={r['candidates']:8d} recall={r['recall']:.4f} {r['seconds']:.2f}s")  # fmt: skip
    return {"records": n, "true_pairs": len(truth), "normalize_seconds": t_norm, "rows": rows}


def missed_by_noise(name: str, k: int) -> dict:
    """Which corruptions cause the true pairs the default blocker misses."""
    records = load_split(name)
    variants = [name_variants(r["name"]) for r in records]
    truth = true_pairs([r["entity_id"] for r in records])
    found = set(tfidf_knn(variants, k=k)) | acronym_blocking(variants)
    missed = truth - found
    tags = Counter()
    for a, b in missed:
        for t in set(records[a]["noise"].split(";") + records[b]["noise"].split(";")) - {""}:
            tags[t] += 1
    examples = [(records[a]["name"], records[b]["name"]) for a, b in sorted(missed)[:25]]
    return {"missed": len(missed), "by_noise": dict(tags.most_common()), "examples": examples}


def choose_k(dev: dict) -> int:
    """Smallest k where doubling k adds less than one point of recall (decided on dev only)."""
    rows = {r["k"]: r["recall"] for r in dev["rows"] if r["method"] == "tfidf_knn+acronym"}
    for k in sorted(rows):
        bigger = [k2 for k2 in rows if k2 >= 2 * k]
        if bigger and rows[min(bigger)] - rows[k] < 0.01:
            return k
    return max(rows)


def main() -> None:
    dev = run_split("dev")
    test = run_split("test")
    k = choose_k(dev)
    print(f"chosen k (on dev) = {k}")
    save_json(
        RESULTS / "blocking.json",
        {"meta": metadata(), "chosen_k": k, "dev": dev, "test": test, "test_missed": missed_by_noise("test", k)},
    )


if __name__ == "__main__":
    main()
