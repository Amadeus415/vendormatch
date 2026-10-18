import numpy as np
import pytest

from vendormatch.metrics import (
    average_precision,
    best_threshold,
    cluster_scores,
    pair_scores,
    pr_curve,
    true_pairs,
)


def test_true_pairs():
    assert true_pairs(["a", "b", "a", "a", "c"]) == {(0, 2), (0, 3), (2, 3)}


def test_pair_scores_counts_missing_pairs_as_false_negatives():
    truth = {(0, 1), (2, 3), (4, 5)}
    s = pair_scores([(0, 1), (2, 4)], truth)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 2)
    assert s["precision"] == 0.5
    assert s["recall"] == pytest.approx(1 / 3)
    assert s["f1"] == pytest.approx(0.4)


def test_pr_curve_recall_is_capped_by_blocking():
    scores = np.array([0.9, 0.8, 0.3])
    labels = np.array([1, 0, 1])
    p, r, t = pr_curve(scores, labels, n_true_total=4)  # one true pair never reached the scorer
    assert list(t) == [0.9, 0.8, 0.3]
    assert list(p) == [1.0, 0.5, pytest.approx(2 / 3)]
    assert list(r) == [0.25, 0.25, 0.5]
    assert average_precision(scores, labels, 4) == pytest.approx(0.25 * 1.0 + 0.25 * (2 / 3))


def test_pr_curve_groups_tied_scores():
    p, r, t = pr_curve(np.array([0.5, 0.5, 0.1]), np.array([1, 0, 1]), 2)
    assert list(t) == [0.5, 0.1]
    assert list(p) == [0.5, pytest.approx(2 / 3)]


def test_best_threshold():
    thr, f1 = best_threshold(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]), 2)
    assert thr == 0.8
    assert f1 == 1.0


def test_cluster_scores_perfect_and_merged():
    truth = ["a", "a", "b", "b", "c"]
    perfect = cluster_scores([0, 0, 1, 1, 2], truth)
    assert perfect["pairwise_f1"] == 1.0 and perfect["bcubed_f1"] == 1.0
    assert perfect["exact_cluster_rate"] == 1.0 and perfect["ari"] == 1.0

    merged = cluster_scores([0, 0, 0, 0, 1], truth)  # a and b glued together
    assert merged["pairwise_recall"] == 1.0
    assert merged["pairwise_precision"] == pytest.approx(2 / 6)
    assert merged["bcubed_recall"] == 1.0
    assert merged["bcubed_precision"] == pytest.approx((4 * 0.5 + 1) / 5)
    assert merged["exact_cluster_rate"] == pytest.approx(1 / 3)


def test_cluster_scores_all_singletons():
    s = cluster_scores([0, 1, 2], ["a", "a", "b"])
    assert s["pairwise_precision"] == 0.0 and s["pairwise_recall"] == 0.0
    assert s["bcubed_precision"] == 1.0
