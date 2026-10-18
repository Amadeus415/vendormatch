"""Stage 3 inputs: cheap per-pair similarity features.

Every feature is a number in [0, 1] (or NaN when a field is missing on either side), computed from the
normalized name variants and the address fields. The learned scorer combines them; the baselines use
one feature each.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from scipy import sparse

from .blocking import Pair, _expand, tfidf_matrix
from .normalize import WEAK, initials, name_variants, normalize_city, normalize_street, normalize_zip

FEATURES = [
    "tfidf_cos",
    "ratio",
    "partial_ratio",
    "token_sort",
    "token_set",
    "jaro_winkler",
    "exact",
    "prefix",
    "acronym",
    "first_token",
    "idf_jaccard",
    "rare_unshared",
    "number_mismatch",
    "len_ratio",
    "state_eq",
    "city_sim",
    "zip_eq",
    "street_sim",
    "any_card",
    "same_source",
]

NUMBERISH = re.compile(r"^(\d+|i{1,3}|iv|v|vi{1,3}|ix|x)$")


@dataclass
class Prepared:
    """Records after normalization, plus corpus statistics the features need."""

    names: list[str]
    variants: list[list[str]]
    city: list[str]
    state: list[str]
    zip: list[str]
    street: list[str]
    source: list[str]
    tfidf: sparse.csr_matrix  # one row per variant
    owner: np.ndarray  # variant row -> record index
    rows_of: list[list[int]]  # record index -> its variant rows
    idf: dict[str, float]

    def __len__(self) -> int:
        return len(self.names)


def prepare(records: Sequence[Mapping[str, str | None]]) -> Prepared:
    variants = [name_variants(r.get("name") or "") for r in records]
    rows, owner = _expand(variants)
    rows_of: list[list[int]] = [[] for _ in records]
    for row, rec in enumerate(owner):
        rows_of[rec].append(row)
    df = Counter(t for names in variants for t in {t for n in names for t in n.split()})
    n = max(len(records), 1)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return Prepared(
        names=[r.get("name") or "" for r in records],
        variants=variants,
        city=[normalize_city(r.get("city")) for r in records],
        state=[(r.get("state") or "").strip().upper() for r in records],
        zip=[normalize_zip(r.get("zip")) for r in records],
        street=[normalize_street(r.get("street")) for r in records],
        source=[(r.get("source") or "").strip().lower() for r in records],
        tfidf=tfidf_matrix(rows),
        owner=owner,
        rows_of=rows_of,
        idf=idf,
    )


def _numbers(tokens: set[str]) -> set[str]:
    return {t for t in tokens if NUMBERISH.match(t)}


def _name_features(a: str, b: str, idf: Mapping[str, float], max_idf: float) -> list[float]:
    ta, tb = a.split(), b.split()
    sa, sb = set(ta), set(tb)
    ca, cb = a.replace(" ", ""), b.replace(" ", "")
    shortest = min(len(ca), len(cb))
    prefix = 1.0 if shortest >= 4 and (ca.startswith(cb) or cb.startswith(ca)) else 0.0
    ia, ib = initials(a), initials(b)
    acronym = 1.0 if (ia and ia == cb) or (ib and ib == ca) else 0.0
    union = sa | sb
    w_union = sum(idf.get(t, 1.0) for t in union) or 1.0
    w_shared = sum(idf.get(t, 1.0) for t in sa & sb)
    unshared = [idf.get(t, 1.0) for t in (sa ^ sb) if t not in WEAK]
    na, nb = _numbers(sa), _numbers(sb)
    return [
        fuzz.ratio(a, b) / 100,
        fuzz.partial_ratio(a, b) / 100,
        fuzz.token_sort_ratio(a, b) / 100,
        fuzz.token_set_ratio(a, b) / 100,
        JaroWinkler.similarity(a, b),
        1.0 if a == b and a else 0.0,
        prefix,
        acronym,
        1.0 if ta and tb and ta[0] == tb[0] else 0.0,
        w_shared / w_union,
        (max(unshared) / max_idf) if unshared else 0.0,
        1.0 if na != nb and (na or nb) else 0.0,
        min(len(ca), len(cb)) / max(len(ca), len(cb), 1),
    ]


def _eq(a: str, b: str) -> float:
    return float("nan") if not a or not b else float(a == b)


def pair_cosines(p: Prepared, pairs: Sequence[Pair]) -> np.ndarray:
    """TF-IDF cosine for each pair (max over name variants), computed in one sparse pass."""
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    left, right, group = [], [], []
    for k, (i, j) in enumerate(pairs):
        for a in p.rows_of[i]:
            for b in p.rows_of[j]:
                left.append(a)
                right.append(b)
                group.append(k)
    dots = np.asarray(p.tfidf[left].multiply(p.tfidf[right]).sum(axis=1)).ravel()
    out = np.zeros(len(pairs), dtype=np.float32)
    np.maximum.at(out, np.asarray(group), dots)
    return out


def pair_features(p: Prepared, pairs: Sequence[Pair]) -> np.ndarray:
    """Feature matrix with one row per pair, columns in FEATURES order."""
    out = np.empty((len(pairs), len(FEATURES)), dtype=np.float32)
    cosines = pair_cosines(p, pairs)
    max_idf = max(p.idf.values(), default=1.0)
    for k, (i, j) in enumerate(pairs):
        best = None
        for a in p.variants[i]:
            for b in p.variants[j]:
                f = _name_features(a, b, p.idf, max_idf)
                best = f if best is None else [max(x, y) for x, y in zip(best, f, strict=True)]
        # Two "mismatch" features should take the min over variants, not the max.
        if len(p.variants[i]) * len(p.variants[j]) > 1:
            mins = [_name_features(a, b, p.idf, max_idf) for a in p.variants[i] for b in p.variants[j]]
            best[10] = min(m[10] for m in mins)
            best[11] = min(m[11] for m in mins)
        city = (
            JaroWinkler.similarity(p.city[i], p.city[j]) if p.city[i] and p.city[j] else float("nan")
        )
        street = (
            fuzz.token_set_ratio(p.street[i], p.street[j]) / 100
            if p.street[i] and p.street[j]
            else float("nan")
        )
        out[k] = [
            float(cosines[k]),
            *best,
            _eq(p.state[i], p.state[j]),
            city,
            _eq(p.zip[i], p.zip[j]),
            street,
            1.0 if "card" in (p.source[i], p.source[j]) else 0.0,
            1.0 if p.source[i] and p.source[i] == p.source[j] else 0.0,
        ]
    return out
