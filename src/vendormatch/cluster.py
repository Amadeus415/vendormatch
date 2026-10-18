"""Stage 5: turn pairwise match decisions into golden records.

union_find   every match edge merges two clusters (transitive closure). Simple, but one bad edge can
             glue two real companies together, and chains (A~B, B~C) pull in records that look nothing alike.
average_link agglomerative merging on the sparse score graph. Two clusters merge only if the average
             score across all record pairs between them clears the threshold, with unscored pairs
             counted as 0. A single bridging edge is not enough to merge two big clusters.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping

import numpy as np

Pair = tuple[int, int]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def relabel(roots: Iterable[int]) -> np.ndarray:
    """Map arbitrary cluster roots to 0..k-1 in first-seen order."""
    seen: dict[int, int] = {}
    return np.asarray([seen.setdefault(r, len(seen)) for r in roots])


def union_find(n: int, edges: Iterable[Pair]) -> np.ndarray:
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    return relabel(uf.find(i) for i in range(n))


def average_link(n: int, scores: Mapping[Pair, float], threshold: float = 0.5) -> np.ndarray:
    """Average-linkage agglomerative clustering over a sparse similarity graph."""
    parent = list(range(n))
    size = [1] * n
    # link[c][d] = sum of scores between clusters c and d (only pairs that were scored contribute)
    link: list[dict[int, float]] = [{} for _ in range(n)]
    for (a, b), s in scores.items():
        if a != b:
            link[a][b] = link[a].get(b, 0.0) + s
            link[b][a] = link[b].get(a, 0.0) + s
    heap = [(-s, a, b) for (a, b), s in scores.items() if s >= threshold and a != b]
    heapq.heapify(heap)
    while heap:
        neg, a, b = heapq.heappop(heap)
        if parent[a] != a or parent[b] != b or b not in link[a]:
            continue  # one side was already merged away
        avg = link[a][b] / (size[a] * size[b])
        if abs(avg + neg) > 1e-9:  # stale entry: the clusters changed since it was pushed
            if avg >= threshold:
                heapq.heappush(heap, (-avg, a, b))
            continue
        keep, gone = (a, b) if len(link[a]) >= len(link[b]) else (b, a)
        parent[gone] = keep
        size[keep] += size[gone]
        del link[keep][gone]
        for c, s in link[gone].items():
            if c != keep:
                link[keep][c] = link[keep].get(c, 0.0) + s
                del link[c][gone]
                link[c][keep] = link[keep][c]
        link[gone] = {}
        for c, s in link[keep].items():
            avg_c = s / (size[keep] * size[c])
            if avg_c >= threshold:
                heapq.heappush(heap, (-avg_c, min(keep, c), max(keep, c)))

    def root(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    return relabel(root(i) for i in range(n))


SOURCE_PRIORITY = {"erp": 0, "crm": 1, "ap": 2, "card": 3}


def golden_names(labels: np.ndarray, names: list[str], sources: list[str]) -> dict[int, str]:
    """Pick one display name per cluster: prefer the vendor master (erp), then the longest name.
    Card descriptors come last because they are truncated."""
    best: dict[int, tuple] = {}
    for i, c in enumerate(labels):
        key = (SOURCE_PRIORITY.get(sources[i], 2), -len(names[i]), names[i])
        if int(c) not in best or key < best[int(c)]:
            best[int(c)] = key
    return {c: k[2] for c, k in best.items()}
