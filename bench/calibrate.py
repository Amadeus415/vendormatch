"""Combine the LLM's verdict with the model's score, fit on dev band pairs only.

Two problems with letting the LLM simply overrule the model inside the band:
- Its probability from the decision token is almost always 0 or 1, and average-linkage clustering reads
  scores as strength of evidence, so one confident mistake can pull a record into the wrong cluster.
- It is not reliably better than the model on these pairs. It catches renames and acronyms the
  model can't, and it over-matches different companies that share a first word.

So the shipped cascade feeds both into a three-parameter logistic regression,
    P(same) = sigmoid(b + w_model * z + w_llm * said_match),   z = logit(model score) - logit(threshold),
fit on every dev pair in the band. Writes src/vendormatch/llm_calibration.json (read by the pipeline) and
results/calibration.json, which compares this against the model alone and the LLM alone on dev, using
5-fold cross-validated predictions so the comparison is not in-sample.

    uv run python -m bench.calibrate
"""

from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from vendormatch.cluster import average_link
from vendormatch.llm import CALIBRATION_PATH
from vendormatch.metrics import cluster_scores, pair_scores, prf, true_pairs
from vendormatch.pipeline import _center

from .common import RESULTS, load_json, load_split, metadata, save_json
from .run_llm import LLM_DIR


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def band_stats(pred: np.ndarray, y: np.ndarray) -> dict:
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return {**prf(tp, fp, fn), "accuracy": float((pred == y).mean())}


def main() -> None:
    sc = load_json(RESULTS / "scorers.json")
    thr = sc["methods"]["gbm"]["threshold"]
    lo, hi = sc["meta"]["band"]
    records = load_split("dev")
    truth_ids = [r["entity_id"] for r in records]
    truth = true_pairs(truth_ids)
    rows = list(csv.DictReader((RESULTS / "dev_pairs.csv").open()))
    pairs = [(int(r["a"]), int(r["b"])) for r in rows]
    gbm = np.array([float(r["gbm"]) for r in rows])
    y_all = np.array([r["label"] == "1" for r in rows])
    verdicts = {}
    for tag in ("dev-prompt-v2", "dev-band-v2"):
        for line in (LLM_DIR / f"{tag}.jsonl").open():
            r = json.loads(line)
            verdicts[(r["a"], r["b"])] = r
    band = np.array([i for i, p in enumerate(pairs) if lo <= gbm[i] <= hi and p in verdicts])
    z = logit(gbm[band]) - logit(np.array([thr]))[0]
    said = np.array([verdicts[pairs[i]]["match"] for i in band], dtype=float)
    X, y = np.c_[z, said], y_all[band]
    print(f"dev band: {len(band)} pairs with an LLM verdict, {int(y.sum())} true matches")

    model = LogisticRegression()
    cv = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    model.fit(X, y)
    b, (w_model, w_llm) = float(model.intercept_[0]), model.coef_[0].tolist()
    print(f"P(same) = sigmoid({b:.3f} + {w_model:.3f} * z + {w_llm:.3f} * said_match)")

    band_comparison = {
        "model_alone": band_stats(gbm[band] >= thr, y),
        "llm_alone": band_stats(said > 0, y),
        "combined_cv": band_stats(cv[:, 1] >= 0.5, y),
    }
    base = _center(gbm, thr)
    variants = {"model_alone": base.copy(), "llm_overrules_raw": base.copy(), "combined_cv": base.copy()}
    for k, i in enumerate(band):
        variants["llm_overrules_raw"][i] = verdicts[pairs[i]]["p_match"]
        variants["combined_cv"][i] = cv[k, 1]
    end_to_end = {}
    for name, s in variants.items():
        labels = average_link(len(records), dict(zip(pairs, s.tolist(), strict=True)))
        end_to_end[name] = {"pair": pair_scores([p for p, v in zip(pairs, s, strict=True) if v >= 0.5], truth),
                            "cluster": cluster_scores(labels, truth_ids)}  # fmt: skip
    for name in variants:
        bs, e = band_comparison.get(name, {}), end_to_end[name]
        band_txt = f"band acc {bs['accuracy']:.3f} F1 {bs['f1']:.3f} | " if bs else ""
        print(f"  {name:18} {band_txt}pair F1 {e['pair']['f1']:.4f}, cluster pairwise F1 "
              f"{e['cluster']['pairwise_f1']:.4f}, B-cubed {e['cluster']['bcubed_f1']:.4f}")  # fmt: skip

    combiner = {"intercept": b, "w_model": w_model, "w_llm": w_llm, "threshold": thr, "dev_band_pairs": int(len(band)),
                "prompt": "v2", "model": "mlx-community/Qwen3-14B-4bit"}  # fmt: skip
    CALIBRATION_PATH.write_text(json.dumps(combiner, indent=2) + "\n")
    print(f"wrote {CALIBRATION_PATH}")
    save_json(RESULTS / "calibration.json", {"meta": metadata(), "combiner": combiner,
                                             "dev_band": band_comparison, "dev_end_to_end": end_to_end})  # fmt: skip


if __name__ == "__main__":
    main()
