from vendormatch.blocking import acronym_blocking, candidates, reduction_ratio, tfidf_knn, token_blocking
from vendormatch.normalize import name_variants

NAMES = [
    "Blue Bottle Coffee Inc",
    "SQ *BLUE BOTTLE COF",
    "International Business Machines Corp",
    "IBM",
    "Delta Air Lines Inc",
    "DELTA AIRLINES",
    "Zebra Technologies",
]
V = [name_variants(n) for n in NAMES]


def test_tfidf_knn_finds_typo_and_truncation_neighbours():
    pairs = tfidf_knn(V, k=1)
    assert (0, 1) in pairs
    assert (4, 5) in pairs
    assert all(a < b for a, b in pairs)
    assert all(0 < s <= 1.0 + 1e-6 for s in pairs.values())


def test_acronym_blocking_links_ibm():
    assert acronym_blocking(V) == {(2, 3)}


def test_token_blocking_respects_max_df():
    pairs = token_blocking(V, max_df=10)
    assert (0, 1) in pairs  # share "blue" and "bottle"
    assert token_blocking(V, max_df=1) == set()


def test_candidates_is_sorted_union():
    pairs = candidates(V, k=2)
    assert pairs == sorted(pairs)
    assert (2, 3) in pairs and (0, 1) in pairs


def test_reduction_ratio():
    assert reduction_ratio(100, 0) == 1.0
    assert reduction_ratio(100, 4950) == 0.0
    assert abs(reduction_ratio(1000, 4995) - 0.99) < 1e-12
