"""The cost curve and the 'LLM on everything' estimate in bench/run_cascade.py."""

import numpy as np
import pytest

from bench.run_cascade import budget_curve, llm_everything_estimate
from vendormatch.metrics import pair_scores


def make(n=40, seed=0):
    rng = np.random.default_rng(seed)
    pairs = [(i, i + 100) for i in range(n)]
    y = rng.integers(0, 2, n)
    gbm = np.clip(y * 0.6 + rng.normal(0.2, 0.25, n), 0, 1)
    truth = {p for p, t in zip(pairs, y, strict=True) if t} | {(500, 501)}  # one pair blocking missed
    d = {"pairs": pairs, "y": y, "gbm": gbm, "thr": 0.5, "truth": truth, "index": {p: i for i, p in enumerate(pairs)}}
    llm = {p: {"match": bool(rng.random() < 0.9) == bool(t), "p_match": 0.0} for p, t in zip(pairs, y, strict=True)}
    order = np.argsort(np.abs(gbm - 0.5))
    return d, llm, order


def test_budget_curve_endpoints():
    d, llm, order = make()
    curve = budget_curve(d, llm, order, covered=len(d["pairs"]))
    gbm_only = pair_scores([p for p, s in zip(d["pairs"], d["gbm"], strict=True) if s >= 0.5], d["truth"])
    llm_only = pair_scores([p for p in d["pairs"] if llm[p]["match"]], d["truth"])
    assert curve[0]["llm_pairs"] == 0 and curve[0]["f1"] == pytest.approx(gbm_only["f1"])
    assert curve[-1]["llm_pairs"] == len(d["pairs"]) and curve[-1]["f1"] == pytest.approx(llm_only["f1"])


def test_estimate_is_exact_when_the_tail_sample_is_the_whole_tail():
    d, llm, order = make()
    head = 15
    tail = {d["pairs"][i]: llm[d["pairs"][i]] for i in order[head:]}
    est = llm_everything_estimate(d, llm, tail, order, covered=head)
    exact = pair_scores([p for p in d["pairs"] if llm[p]["match"]], d["truth"])
    assert est["f1"] == pytest.approx(exact["f1"])
    assert est["precision"] == pytest.approx(exact["precision"])
    assert est["recall"] == pytest.approx(exact["recall"])


def test_estimate_scales_a_partial_sample():
    d, llm, order = make(n=400, seed=1)
    head = 100
    rest = list(order[head:])
    sample = {d["pairs"][i]: llm[d["pairs"][i]] for i in rest[::2]}  # half the tail
    est = llm_everything_estimate(d, llm, sample, order, covered=head)
    exact = pair_scores([p for p in d["pairs"] if llm[p]["match"]], d["truth"])
    assert est["tail_sample"] == len(sample)
    assert abs(est["f1"] - exact["f1"]) < 0.05
    assert est["f1_ci95"][0] <= est["f1"] <= est["f1_ci95"][1]
