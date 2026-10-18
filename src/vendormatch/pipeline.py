"""The cascade, end to end: normalize -> block -> score -> (LLM on the unsure band) -> cluster."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .blocking import Pair, candidates
from .cluster import average_link, golden_names, union_find
from .features import pair_features, prepare
from .scorer import PairScorer


@dataclass
class Result:
    clusters: np.ndarray  # cluster id per input record
    golden: dict[int, str]  # cluster id -> display name
    pairs: list[Pair]  # candidate pairs that were scored
    scores: np.ndarray  # final match probability per candidate pair
    decided_by: list[str]  # "model" or "llm" per candidate pair
    reasons: dict[Pair, str] = field(default_factory=dict)  # LLM reasons, for pairs it judged
    seconds: dict[str, float] = field(default_factory=dict)  # wall time per stage

    def matches(self, threshold: float = 0.5) -> list[Pair]:
        return [p for p, s in zip(self.pairs, self.scores, strict=True) if s >= threshold]


def dedupe(
    records: Sequence[Mapping[str, str | None]],
    scorer: PairScorer | None = None,
    llm=None,
    k: int = 10,
    clustering: str = "average_link",
) -> Result:
    """Group records that refer to the same company.

    records    dicts with at least "name"; optional "street", "city", "state", "zip", "source"
               (source is one of erp, ap, card, crm; it helps but is not required)
    scorer     a trained PairScorer; defaults to the model shipped with the package
    llm        an LLMJudge; when given, pairs whose score falls in scorer.band go to it
    """
    scorer = scorer or PairScorer.load()
    t: dict[str, float] = {}

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
    scores = scorer.predict(X)
    t["score"] = time.perf_counter() - t0

    # Rescale so 0.5 is the decision boundary for every stage that follows.
    final = _center(scores, scorer.threshold)
    decided_by = ["model"] * len(pairs)
    reasons: dict[Pair, str] = {}
    if llm is not None:
        t0 = time.perf_counter()
        lo, hi = scorer.band
        for n, (pair, s) in enumerate(zip(pairs, scores, strict=True)):
            if lo <= s <= hi:
                j = llm.judge(records[pair[0]], records[pair[1]])
                final[n] = j.p_match
                decided_by[n] = "llm"
                reasons[pair] = j.reason
        t["llm"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if clustering == "union_find":
        labels = union_find(len(records), [p for p, s in zip(pairs, final, strict=True) if s >= 0.5])
    else:
        labels = average_link(len(records), dict(zip(pairs, final.tolist(), strict=True)), threshold=0.5)
    t["cluster"] = time.perf_counter() - t0

    golden = golden_names(labels, prep.names, prep.source)
    return Result(labels, golden, pairs, final, decided_by, reasons, t)


def _center(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Piecewise-linear map that sends `threshold` to 0.5 and keeps 0 and 1 fixed."""
    s = np.asarray(scores, dtype=np.float64)
    if not 0 < threshold < 1:
        return s
    return np.where(s < threshold, 0.5 * s / threshold, 0.5 + 0.5 * (s - threshold) / (1 - threshold))
