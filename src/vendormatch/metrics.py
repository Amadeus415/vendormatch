"""Pair-level and cluster-level accuracy metrics.

Pair recall is always measured against every true pair in the data, including pairs the blocker never
proposed. Otherwise a blocker that drops hard pairs would make every later stage look better.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

Pair = tuple[int, int]


def true_pairs(entity_ids: Sequence) -> set[Pair]:
    groups: dict = defaultdict(list)
    for i, e in enumerate(entity_ids):
        groups[e].append(i)
    out: set[Pair] = set()
    for ids in groups.values():
        out.update((a, b) for x, a in enumerate(ids) for b in ids[x + 1 :])
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def pair_scores(predicted: Iterable[Pair], truth: set[Pair]) -> dict:
    pred = set(predicted)
    tp = len(pred & truth)
    return prf(tp, len(pred) - tp, len(truth) - tp)


def pr_curve(scores: np.ndarray, labels: np.ndarray, n_true_total: int) -> tuple[np.ndarray, ...]:
    """Precision and recall at every threshold. Recall uses n_true_total (all true pairs), so the
    curve tops out at the blocker's recall instead of 1.0."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order].astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # one point per distinct threshold
    tp, fp, thr = tp[last], fp[last], s[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_true_total, 1)
    return precision, recall, thr


def best_threshold(scores: np.ndarray, labels: np.ndarray, n_true_total: int) -> tuple[float, float]:
    """Threshold that maximizes F1 (pairs with score >= threshold are matches)."""
    precision, recall, thr = pr_curve(scores, labels, n_true_total)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    k = int(np.argmax(f1))
    return float(thr[k]), float(f1[k])


def average_precision(scores: np.ndarray, labels: np.ndarray, n_true_total: int) -> float:
    precision, recall, _ = pr_curve(scores, labels, n_true_total)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _contingency(pred: Sequence, truth: Sequence) -> tuple[Counter, Counter, Counter]:
    cells = Counter(zip(pred, truth, strict=True))
    return cells, Counter(pred), Counter(truth)


def cluster_scores(pred: Sequence, truth: Sequence) -> dict:
    """Metrics for a full clustering (one label per record).

    pairwise_*  precision/recall/F1 over all within-cluster pairs implied by the clustering
    bcubed_*    per-record precision/recall, averaged (less dominated by big clusters)
    exact       share of true entities recovered exactly: same records, nothing extra
    """
    n = len(pred)
    cells, a, b = _contingency(pred, truth)
    c2 = lambda x: x * (x - 1) // 2  # noqa: E731
    tp = sum(c2(v) for v in cells.values())
    pred_pairs = sum(c2(v) for v in a.values())
    true_pairs_n = sum(c2(v) for v in b.values())
    pairwise = prf(tp, pred_pairs - tp, true_pairs_n - tp)
    bp = sum(v * v / a[p] for (p, _), v in cells.items()) / n if n else 0.0
    br = sum(v * v / b[t] for (_, t), v in cells.items()) / n if n else 0.0
    bf = 2 * bp * br / (bp + br) if bp + br else 0.0
    exact = sum(1 for (p, t), v in cells.items() if v == a[p] == b[t])
    return {
        "pairwise_precision": pairwise["precision"],
        "pairwise_recall": pairwise["recall"],
        "pairwise_f1": pairwise["f1"],
        "bcubed_precision": bp,
        "bcubed_recall": br,
        "bcubed_f1": bf,
        "exact_cluster_rate": exact / len(b) if b else 0.0,
        "ari": float(adjusted_rand_score(truth, pred)) if n else 0.0,
        "n_pred_clusters": len(a),
        "n_true_clusters": len(b),
    }
