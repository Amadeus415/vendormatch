"""Stage 2: candidate generation. Comparing every record with every other is n^2/2 pairs, so first
pick a small set of pairs worth scoring. Recall lost here is lost for good, so it is measured on its own.

Three blockers, all returning sets of (i, j) record-index pairs with i < j:
  tfidf_knn  character 3-gram TF-IDF, k nearest neighbours by cosine (catches typos, truncation, suffixes)
  token      records that share a rare word (classic token blocking)
  acronym    "ibm" next to "intl business machines" (no character overlap, so the other two miss it)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import WEAK, initials

Pair = tuple[int, int]


def _expand(variants: Sequence[Sequence[str]]) -> tuple[list[str], np.ndarray]:
    """Flatten per-record name variants into rows, remembering which record each row came from."""
    rows, owner = [], []
    for i, names in enumerate(variants):
        for name in names:
            rows.append(name)
            owner.append(i)
    return rows, np.asarray(owner)


def tfidf_matrix(names: Sequence[str]) -> sparse.csr_matrix:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32)
    return vec.fit_transform([n if n else " " for n in names]).tocsr()


def tfidf_knn(variants: Sequence[Sequence[str]], k: int = 10, min_sim: float = 0.0) -> dict[Pair, float]:
    """For every name variant, its k most similar other variants. Returns {(i, j): cosine}, i < j."""
    rows, owner = _expand(variants)
    X = tfidf_matrix(rows)
    XT = X.T.tocsr()
    n = X.shape[0]
    kk = min(k + 1, n)
    chunk = max(64, int(2e7 // max(n, 1)))  # keep each dense block near 80 MB
    lo_all, hi_all, sim_all = [], [], []
    for start in range(0, n, chunk):
        sims = (X[start : start + chunk] @ XT).toarray()
        idx = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
        s = np.take_along_axis(sims, idx, axis=1).ravel()
        a = np.repeat(owner[start : start + sims.shape[0]], kk)
        b = owner[idx.ravel()]
        keep = (a != b) & (s > max(min_sim, 0.0))
        a, b, s = a[keep], b[keep], s[keep]
        lo_all.append(np.minimum(a, b))
        hi_all.append(np.maximum(a, b))
        sim_all.append(s)
    if not lo_all:
        return {}
    lo, hi, sim = np.concatenate(lo_all), np.concatenate(hi_all), np.concatenate(sim_all)
    order = np.lexsort((-sim, hi, lo))  # best similarity first within each (lo, hi)
    lo, hi, sim = lo[order], hi[order], sim[order]
    first = np.ones(len(lo), dtype=bool)
    first[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    return {(int(a), int(b)): float(c) for a, b, c in zip(lo[first], hi[first], sim[first], strict=True)}


def token_blocking(variants: Sequence[Sequence[str]], max_df: int = 50) -> set[Pair]:
    """Pairs that share at least one word that appears in at most max_df records."""
    index: dict[str, set[int]] = defaultdict(set)
    for i, names in enumerate(variants):
        for name in names:
            for tok in name.split():
                if tok not in WEAK and len(tok) > 1:
                    index[tok].add(i)
    pairs: set[Pair] = set()
    for members in index.values():
        if 1 < len(members) <= max_df:
            ids = sorted(members)
            pairs.update((a, b) for x, a in enumerate(ids) for b in ids[x + 1 :])
    return pairs


def acronym_blocking(variants: Sequence[Sequence[str]]) -> set[Pair]:
    """Pair a short single-token name with every multi-word name whose initials spell it."""
    by_initials: dict[str, set[int]] = defaultdict(set)
    short: dict[str, set[int]] = defaultdict(set)
    for i, names in enumerate(variants):
        for name in names:
            ini = initials(name)
            if len(ini) >= 2:
                by_initials[ini].add(i)
            compact = name.replace(" ", "")
            if 2 <= len(compact) <= 6 and compact.isalnum() and (" " not in name or len(name.split()) <= 3):
                short[compact].add(i)
    pairs: set[Pair] = set()
    for key, ids in short.items():
        for a in ids:
            for b in by_initials.get(key, ()):
                if a != b:
                    pairs.add((a, b) if a < b else (b, a))
    return pairs


def candidates(variants: Sequence[Sequence[str]], k: int = 10) -> list[Pair]:
    """The default blocker: TF-IDF kNN plus acronym pairs, sorted for determinism."""
    pairs = set(tfidf_knn(variants, k=k)) | acronym_blocking(variants)
    return sorted(pairs)


def reduction_ratio(n_records: int, n_candidates: int) -> float:
    total = n_records * (n_records - 1) / 2
    return 1.0 - n_candidates / total if total else 0.0
