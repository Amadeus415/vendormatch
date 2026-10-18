"""Stage 3 benchmark: cheap scorers on the blocked candidate pairs, and the shipped model.

Every threshold is chosen on dev. The learned models pick their threshold from 5-fold out-of-fold
predictions on dev (folds split by company), so the threshold is not fit to rows the model memorized.
Then each method is applied once to test.

    uv run python -m bench.run_scorers
"""

from __future__ import annotations

import csv
import math
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from vendormatch.blocking import candidates
from vendormatch.features import FEATURES, pair_features, prepare
from vendormatch.metrics import average_precision, best_threshold, pair_scores, pr_curve, true_pairs
from vendormatch.scorer import MODEL_PATH, PairScorer

from .common import RESULTS, load_json, load_split, metadata, save_json

SINGLE_FEATURE = {
    "exact": "exact",
    "tfidf_cosine": "tfidf_cos",
    "rapidfuzz_ratio": "ratio",
    "rapidfuzz_partial": "partial_ratio",
    "rapidfuzz_token_sort": "token_sort",
    "rapidfuzz_token_set": "token_set",
    "jaro_winkler": "jaro_winkler",
}
# Pairs whose GBM odds are within 20:1 of its own threshold (|log-odds distance| <= 3) go to the LLM.
# A prior, not tuned on anything: it is the same rule as "most uncertain first" in the cost curve.
BAND_LOGIT = 3.0


def band_around(threshold: float, width: float = BAND_LOGIT) -> tuple[float, float]:
    logit = math.log(threshold / (1 - threshold))
    sigmoid = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    return sigmoid(logit - width), sigmoid(logit + width)


def build(name: str, k: int) -> dict:
    records = load_split(name)
    t0 = time.perf_counter()
    prep = prepare(records)
    t_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    pairs = candidates(prep.variants, k=k)
    t_block = time.perf_counter() - t0
    t0 = time.perf_counter()
    X = pair_features(prep, pairs)
    t_feat = time.perf_counter() - t0
    truth = true_pairs([r["entity_id"] for r in records])
    y = np.array([p in truth for p in pairs], dtype=np.int8)
    print(f"{name}: {len(records)} records, {len(pairs)} candidates, {int(y.sum())}/{len(truth)} true pairs kept, "
          f"features {t_feat:.1f}s ({len(pairs) / t_feat:,.0f} pairs/s)")  # fmt: skip
    return {"records": records, "pairs": pairs, "X": X, "y": y, "truth": truth,
            "seconds": {"normalize": t_prep, "block": t_block, "features": t_feat}}  # fmt: skip


def oof_scores(kind: str, X: np.ndarray, y: np.ndarray, groups: list[str], seed: int = 0) -> np.ndarray:
    """5-fold out-of-fold scores, folds split by company so no company is on both sides of a fold."""
    out = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        out[va] = PairScorer(kind).fit(X[tr], y[tr], seed=seed).predict(X[va])
    return out


def curve_points(scores, labels, n_true, n_points=200) -> dict:
    p, r, t = pr_curve(scores, labels, n_true)
    idx = np.unique(np.linspace(0, len(p) - 1, min(n_points, len(p))).astype(int))
    return {"precision": p[idx], "recall": r[idx], "threshold": t[idx]}


def threshold_sweep(scores, labels, n_true) -> dict:
    grid = np.round(np.linspace(0.02, 0.98, 49), 3)
    rows = []
    for th in grid:
        m = scores >= th
        tp = int((labels[m] == 1).sum())
        fp = int(m.sum()) - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(n_true, 1)
        rows.append({"threshold": float(th), "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)})
    return rows


def evaluate(name: str, s_dev, s_test, dev, test, threshold=None) -> dict:
    if threshold is None:
        threshold, _ = best_threshold(s_dev, dev["y"], len(dev["truth"]))
    pred = [p for p, s in zip(test["pairs"], s_test, strict=True) if s >= threshold]
    res = pair_scores(pred, test["truth"])
    res.update(
        method=name,
        threshold=float(threshold),
        average_precision=average_precision(s_test, test["y"], len(test["truth"])),
        pr_curve=curve_points(s_test, test["y"], len(test["truth"])),
        sweep_test=threshold_sweep(s_test, test["y"], len(test["truth"])),
    )
    print(f"  {name:22} thr={threshold:.3f}  P={res['precision']:.4f} R={res['recall']:.4f} F1={res['f1']:.4f} "
          f"AP={res['average_precision']:.4f}")  # fmt: skip
    return res


def main() -> None:
    k = load_json(RESULTS / "blocking.json")["chosen_k"]
    dev, test = build("dev", k), build("test", k)
    col = {f: i for i, f in enumerate(FEATURES)}
    methods = {}
    for name, feat in SINGLE_FEATURE.items():
        s_dev, s_test = dev["X"][:, col[feat]], test["X"][:, col[feat]]
        methods[name] = evaluate(name, s_dev, s_test, dev, test, threshold=0.5 if name == "exact" else None)

    learned = {}
    for kind in ("logreg", "gbm"):
        t0 = time.perf_counter()
        groups = [dev["records"][a]["entity_id"] for a, _ in dev["pairs"]]  # company of the first record
        oof = oof_scores(kind, dev["X"], dev["y"], groups)
        thr, _ = best_threshold(oof, dev["y"], len(dev["truth"]))
        model = PairScorer(kind, threshold=thr, band=band_around(thr)).fit(dev["X"], dev["y"])
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        s_test = model.predict(test["X"])
        t_pred = time.perf_counter() - t0
        methods[kind] = evaluate(kind, oof, s_test, dev, test, threshold=thr)
        methods[kind]["train_seconds"] = t_train
        methods[kind]["predict_pairs_per_second"] = len(s_test) / t_pred
        learned[kind] = (model, s_test, oof)

    gbm, s_gbm, oof_gbm = learned["gbm"]
    gbm.meta = {"trained_on": "data/dev.csv", "k": k, "dev_pairs": len(dev["y"]), **metadata()}
    gbm.save(MODEL_PATH)
    importances = permutation_importance(gbm, test)

    # Out-of-fold GBM scores on dev: the LLM prompt was developed on dev pairs in the same band.
    with (RESULTS / "dev_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "label", "gbm"])
        for (a, b), y, sg in zip(dev["pairs"], dev["y"], oof_gbm, strict=True):
            w.writerow([a, b, int(y), f"{sg:.6f}"])

    # Per-pair scores on test, used by the LLM and cascade stages.
    with (RESULTS / "test_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "label", "gbm", "logreg", *FEATURES])
        for (a, b), y, sg, sl, x in zip(test["pairs"], test["y"], s_gbm, learned["logreg"][1], test["X"],
                                        strict=True):  # fmt: skip
            w.writerow([a, b, int(y), f"{sg:.6f}", f"{sl:.6f}", *(f"{v:.4f}" for v in x)])

    save_json(
        RESULTS / "scorers.json",
        {
            "meta": metadata(k=k, band=gbm.band, band_logit=BAND_LOGIT),
            "dev": {"records": len(dev["records"]), "candidates": len(dev["pairs"]), "true_pairs": len(dev["truth"]),
                    "true_in_candidates": int(dev["y"].sum())},  # fmt: skip
            "test": {"records": len(test["records"]), "candidates": len(test["pairs"]),
                     "true_pairs": len(test["truth"]), "true_in_candidates": int(test["y"].sum()),
                     "seconds": test["seconds"],
                     "feature_pairs_per_second": len(test["pairs"]) / test["seconds"]["features"]},  # fmt: skip
            "methods": methods,
            "gbm_permutation_importance": importances,
        },
    )


def permutation_importance(model: PairScorer, test: dict, seed: int = 0) -> dict:
    """Drop in test average precision when one feature column is shuffled."""
    rng = np.random.default_rng(seed)
    base = average_precision(model.predict(test["X"]), test["y"], len(test["truth"]))
    out = {}
    for i, f in enumerate(FEATURES):
        X = test["X"].copy()
        X[:, i] = X[rng.permutation(len(X)), i]
        out[f] = base - average_precision(model.predict(X), test["y"], len(test["truth"]))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
