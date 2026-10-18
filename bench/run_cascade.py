"""Stages 4 and 5 together: how much does the LLM add, at what cost, and what do clusters look like?

Reads the GBM scores (results/test_pairs.csv) and the LLM runs in results/llm/, writes results/cascade.json.
No GPU needed; the LLM outputs are already on disk.

The cascade sends the pairs the GBM is least sure about (smallest distance from its threshold, in
log-odds) to the LLM, and keeps the GBM's decision for everything else. Sweeping how many pairs go to
the LLM gives the cost/accuracy curve.

"LLM on everything" needs a verdict for every candidate pair. The LLM run covers the most uncertain
pairs exhaustively and the rest through a uniform random sample, so that point is an estimate:
the sampled part is scaled up to the stratum size, with a bootstrap interval.

    uv run python -m bench.run_cascade
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vendormatch.cluster import average_link, union_find
from vendormatch.features import FEATURES
from vendormatch.llm import combined_score
from vendormatch.metrics import cluster_scores, pair_scores, prf, true_pairs
from vendormatch.pipeline import _center

from .common import RESULTS, load_json, load_split, metadata, save_json
from .run_llm import LLM_DIR, uncertainty

BUDGETS = [0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
NOISE_TAGS = ["suffix", "punctuation", "abbreviation", "typo", "card_descriptor", "processor_prefix", "store_number",
              "truncated", "charge_city", "attention_line", "remit_address", "dba", "alias", "acronym",
              "former_name"]  # fmt: skip


def read_jsonl(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    return {(r["a"], r["b"]): r for r in map(json.loads, path.open())}


def load() -> dict:
    rows = list(csv.DictReader((RESULTS / "test_pairs.csv").open()))
    sc = load_json(RESULTS / "scorers.json")
    records = load_split("test")
    pairs = [(int(r["a"]), int(r["b"])) for r in rows]
    return {
        "records": records,
        "pairs": pairs,
        "index": {p: i for i, p in enumerate(pairs)},
        "y": np.array([int(r["label"]) for r in rows]),
        "gbm": np.array([float(r["gbm"]) for r in rows]),
        "feat": {f: np.array([float(r[f]) if r[f] != "nan" else np.nan for r in rows]) for f in FEATURES},
        "truth": true_pairs([r["entity_id"] for r in records]),
        "scorers": sc,
        "thr": sc["methods"]["gbm"]["threshold"],
        "band": tuple(sc["meta"]["band"]),
    }


def decisions_scores(d, chosen: np.ndarray) -> dict:
    """Pair-level metrics for a boolean decision per candidate pair."""
    pred = [p for p, m in zip(d["pairs"], chosen, strict=True) if m]
    return pair_scores(pred, d["truth"])


def budget_curve(d, llm: dict, order: np.ndarray, covered: int, extra: tuple[int, ...] = ()) -> list[dict]:
    """Cascade F1 when the first n pairs (in uncertainty order) are decided by the LLM, for every
    budget in BUDGETS (and any extra pair counts) that the LLM run covers."""
    base = d["gbm"] >= d["thr"]
    llm_match = np.array([llm[p]["match"] if p in llm else False for p in d["pairs"]])
    N = len(d["pairs"])
    counts = sorted({int(round(f * N)) for f in BUDGETS} | set(extra) | ({covered} if 0 < covered < N else set()))
    out = []
    for n in counts:
        if n > covered:
            break
        chosen = base.copy()
        chosen[order[:n]] = llm_match[order[:n]]
        out.append({"fraction": n / N, "llm_pairs": n, **decisions_scores(d, chosen)})
    return out


def llm_everything_estimate(d, llm: dict, tail: dict, order: np.ndarray, covered: int, seed: int = 0) -> dict | None:
    """F1 of 'LLM decides every candidate pair', estimated from the exhaustive head plus a random tail sample."""
    N = len(d["pairs"])
    rest = order[covered:]
    rest_set = set(rest.tolist())
    tail_idx = np.array([d["index"][p] for p in tail if d["index"].get(p) in rest_set], dtype=int)
    if covered < N and len(tail_idx) == 0:
        return None
    y = d["y"]
    head = order[:covered]
    head_pred = np.array([llm[d["pairs"][i]]["match"] for i in head], dtype=bool)
    tp_h = int((head_pred & (y[head] == 1)).sum())
    fp_h = int((head_pred & (y[head] == 0)).sum())
    pos_cand = int(y.sum())
    missed_by_blocking = len(d["truth"]) - pos_cand

    def estimate(sample_idx):
        if len(rest) == 0:
            tp, fp = tp_h, fp_h
        else:
            pred = np.array([tail[d["pairs"][i]]["match"] for i in sample_idx], dtype=bool)
            w = len(rest) / len(sample_idx)
            tp = tp_h + w * (pred & (y[sample_idx] == 1)).sum()
            fp = fp_h + w * (pred & (y[sample_idx] == 0)).sum()
        fn = pos_cand - tp + missed_by_blocking
        p, r = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    p, r, f1 = estimate(tail_idx)
    rng = np.random.default_rng(seed)
    boots = [estimate(rng.choice(tail_idx, len(tail_idx))) [2] for _ in range(1000)] if len(rest) else [f1]
    return {"precision": p, "recall": r, "f1": f1, "f1_ci95": [float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))], "exhaustive_pairs": covered, "tail_pairs": len(rest),
            "tail_sample": int(len(tail_idx))}  # fmt: skip


def method_scores(d, llm: dict) -> dict[str, np.ndarray]:
    """A score per candidate pair for each method, centered so 0.5 is that method's tuned threshold."""
    m = d["scorers"]["methods"]
    out = {}
    for name, feat in [("exact", "exact"), ("tfidf_cosine", "tfidf_cos"), ("rapidfuzz_token_set", "token_set"),
                       ("jaro_winkler", "jaro_winkler")]:  # fmt: skip
        out[name] = _center(d["feat"][feat], m[name]["threshold"])
    out["gbm"] = _center(d["gbm"], d["thr"])
    if llm:
        lo, hi = d["band"]
        raw, cal = out["gbm"].copy(), out["gbm"].copy()
        for i, p in enumerate(d["pairs"]):
            if lo <= d["gbm"][i] <= hi and p in llm:
                raw[i] = llm[p]["p_match"]
                c = combined_score(float(d["gbm"][i]), llm[p]["match"])
                cal[i] = raw[i] if c is None else c
        out["cascade"] = cal  # what the pipeline ships: verdict and model score combined (fit on dev)
        out["cascade_raw"] = raw  # the LLM overrules the model with its own probability
    return out


def clustering(d, scores: dict[str, np.ndarray]) -> dict:
    n = len(d["records"])
    truth = [r["entity_id"] for r in d["records"]]
    out = {}
    for name, s in scores.items():
        edges = [p for p, v in zip(d["pairs"], s, strict=True) if v >= 0.5]
        out[name] = {
            "union_find": cluster_scores(union_find(n, edges), truth),
            "average_link": cluster_scores(average_link(n, dict(zip(d["pairs"], s.tolist(), strict=True))), truth),
        }
    return out


def recall_by_noise(d, scores: dict[str, np.ndarray]) -> dict:
    """For true pairs where either record carries a corruption, the share each method finds."""
    tags_of = [set(r["noise"].split(";")) - {""} for r in d["records"]]
    found = {name: {p for p, v in zip(d["pairs"], s, strict=True) if v >= 0.5} for name, s in scores.items()}
    in_cand = set(d["pairs"])
    out = {}
    for tag in NOISE_TAGS:
        group = [p for p in d["truth"] if tag in tags_of[p[0]] or tag in tags_of[p[1]]]
        if not group:
            continue
        out[tag] = {"true_pairs": len(group), "blocking": sum(p in in_cand for p in group) / len(group)}
        for name, f in found.items():
            out[tag][name] = sum(p in f for p in group) / len(group)
    return out


def hard_negatives(d, scores: dict[str, np.ndarray]) -> dict:
    """Different companies whose normalized names look alike (token-set similarity >= 0.8)."""
    mask = (d["y"] == 0) & (d["feat"]["token_set"] >= 0.8)
    out = {"pairs": int(mask.sum())}
    for name, s in scores.items():
        out[name] = {"false_match_rate": float((s[mask] >= 0.5).mean()) if mask.any() else 0.0}
    return out


def band_comparison(d, band_llm: dict) -> dict:
    """Inside the uncertain band only: GBM alone vs the LLM alone, on the same pairs."""
    idx = [d["index"][p] for p in band_llm if p in d["index"]]
    if not idx:
        return {}
    y = d["y"][idx]
    gbm = d["gbm"][idx] >= d["thr"]
    llm = np.array([band_llm[d["pairs"][i]]["match"] for i in idx])

    def stats(pred):
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        return {**prf(tp, fp, fn), "accuracy": float((pred == (y == 1)).mean())}

    return {"pairs": len(idx), "positives": int(y.sum()), "gbm": stats(gbm), "llm": stats(llm)}


def examples(d, scores: dict[str, np.ndarray], band_llm: dict, reasons: dict, k: int = 12) -> dict:
    """Real test pairs each method gets wrong, sampled with a fixed seed. LLM examples come from the
    150 band pairs that were also judged with a written reason."""
    recs = d["records"]
    rng = np.random.default_rng(0)

    def show(i):
        a, b = d["pairs"][i]
        r = reasons.get(d["pairs"][i]) or band_llm.get(d["pairs"][i], {})
        return {"a": {f: recs[a][f] for f in ("source", "name", "city", "state")},
                "b": {f: recs[b][f] for f in ("source", "name", "city", "state")},
                "same_company": bool(d["y"][i]), "gbm": round(float(d["gbm"][i]), 3),
                "jaro_winkler": round(float(d["feat"]["jaro_winkler"][i]), 3),
                "llm_p": round(r["p_match"], 3) if r else None, "reason": r.get("reason", ""),
                "noise": sorted(set(recs[a]["noise"].split(";") + recs[b]["noise"].split(";")) - {""})}  # fmt: skip

    def pick(idx):
        idx = list(idx)
        return [show(i) for i in sorted(rng.choice(idx, min(k, len(idx)), replace=False))] if idx else []

    y = d["y"] == 1
    jw = scores["jaro_winkler"] >= 0.5
    gbm = d["gbm"] >= d["thr"]
    lo, hi = d["band"]
    in_band = (d["gbm"] >= lo) & (d["gbm"] <= hi)
    out = {
        "jaro_winkler_false_match_gbm_right": pick(np.nonzero(jw & ~y & ~gbm)[0]),
        "jaro_winkler_missed_gbm_right": pick(np.nonzero(~jw & y & gbm)[0]),
        "gbm_confident_false_match": pick(np.nonzero(gbm & ~y & ~in_band)[0]),
        "gbm_confident_miss": pick(np.nonzero(~gbm & y & ~in_band)[0]),
    }
    fixed, broken, both_wrong = [], [], []
    for p, r in reasons.items():
        i = d["index"].get(p)
        if i is None:
            continue
        truth, g, m = bool(d["y"][i]), bool(gbm[i]), bool(r["match"])
        if g != truth and m == truth:
            fixed.append(i)
        elif g == truth and m != truth:
            broken.append(i)
        elif g != truth and m != truth:
            both_wrong.append(i)
    band_idx = [d["index"][p] for p in band_llm if p in d["index"]]
    llm_right = np.array([band_llm[d["pairs"][i]]["match"] == y[i] for i in band_idx], dtype=bool)
    gbm_right = np.array([gbm[i] == y[i] for i in band_idx], dtype=bool)
    out.update(counts={"band_llm_fixed": int((llm_right & ~gbm_right).sum()),
                       "band_llm_broke": int((~llm_right & gbm_right).sum()),
                       "band_both_wrong": int((~llm_right & ~gbm_right).sum()),
                       "gbm_confident_errors": int(((gbm != y) & ~in_band).sum())},
               llm_fixed=pick(fixed), llm_broke=pick(broken), both_wrong=pick(both_wrong))  # fmt: skip
    return out


def gbm_band_accuracy(d) -> dict:
    """The GBM's accuracy inside the uncertain band: dev (out-of-fold, folds split by company) vs test."""
    lo, hi = d["band"]
    dev = [r for r in csv.DictReader((RESULTS / "dev_pairs.csv").open()) if lo <= float(r["gbm"]) <= hi]
    dev_acc = float(np.mean([(float(r["gbm"]) >= d["thr"]) == (r["label"] == "1") for r in dev]))
    m = (d["gbm"] >= lo) & (d["gbm"] <= hi)
    test_acc = float(((d["gbm"][m] >= d["thr"]) == (d["y"][m] == 1)).mean())
    return {"dev_pairs": len(dev), "dev_out_of_fold": dev_acc, "test_pairs": int(m.sum()), "test": test_acc}


def prompt_dev() -> dict:
    """Prompt v1 vs v2 on the same dev pairs, next to the GBM's out-of-fold scores on those pairs."""
    runs = {t: read_jsonl(LLM_DIR / f"dev-prompt-{t}.jsonl") for t in ("v1", "v2")}
    if not all(runs.values()):
        return {}
    oof = {(int(r["a"]), int(r["b"])): float(r["gbm"]) for r in csv.DictReader((RESULTS / "dev_pairs.csv").open())}
    thr = load_json(RESULTS / "scorers.json")["methods"]["gbm"]["threshold"]
    common = sorted(set(runs["v1"]) & set(runs["v2"]) & set(oof))
    y = np.array([runs["v1"][p]["label"] for p in common]) == 1

    def stats(pred):
        pred = np.asarray(pred, dtype=bool)
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        return {**prf(tp, fp, fn), "accuracy": float((pred == y).mean())}

    return {"pairs": len(common), "positives": int(y.sum()),
            "gbm_out_of_fold": stats([oof[p] >= thr for p in common]),
            **{f"llm_{t}": stats([runs[t][p]["match"] for p in common]) for t in runs},
            "llm_v2_averaged_with_gbm": stats([(runs["v2"][p]["p_match"] + float(_center([oof[p]], thr)[0])) / 2 >= 0.5
                                               for p in common])}  # fmt: skip


def llm_cost(tag: str) -> dict | None:
    rows = read_jsonl(LLM_DIR / f"{tag}.jsonl")
    meta_path = LLM_DIR / f"{tag}.meta.json"
    if not rows or not meta_path.exists():
        return None
    meta = load_json(meta_path)
    secs = np.array([r["seconds"] for r in rows.values()])
    runs = meta["runs"]
    wall = sum(r["seconds"] for r in runs)
    return {"model": meta["model"], "pairs": len(rows), "reason": meta["reason"],
            "thinking_budget": meta["thinking_budget"],
            "mean_seconds_per_pair": float(secs.mean()),
            "p50_seconds": float(np.median(secs)), "p95_seconds": float(np.percentile(secs, 95)),
            "pairs_per_second": sum(r["pairs"] for r in runs) / wall if wall else None,
            "mean_prefill_tokens": float(np.mean([r["prefill_tokens"] for r in rows.values()])),
            "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in rows.values()]))}  # fmt: skip


def main() -> None:
    d = load()
    N = len(d["pairs"])
    order = np.array(sorted(range(N), key=lambda i: (uncertainty(d["gbm"][i], d["thr"]), d["pairs"][i])))
    band = read_jsonl(LLM_DIR / "qwen3-14b_band.jsonl")
    reasons = read_jsonl(LLM_DIR / "qwen3-14b_reasons.jsonl")
    llm_all = {**band, **read_jsonl(LLM_DIR / "qwen3-14b_all.jsonl")}  # the head run skips pairs the band run judged
    tail = read_jsonl(LLM_DIR / "qwen3-14b_tail.jsonl")
    # Head coverage: how many pairs, in uncertainty order, have an LLM verdict with no gaps.
    covered = 0
    while covered < N and d["pairs"][order[covered]] in llm_all:
        covered += 1
    lo, hi = d["band"]
    in_band = (d["gbm"] >= lo) & (d["gbm"] <= hi)
    print(f"{N} test candidate pairs; LLM head covers {covered} ({covered / N:.1%}); band holds {in_band.sum()} "
          f"({in_band.mean():.1%}); tail sample {len(tail)}")  # fmt: skip

    curve = budget_curve(d, llm_all, order, covered, extra=(int(in_band.sum()),))
    everything = llm_everything_estimate(d, llm_all, tail, order, covered)
    verdicts = band or llm_all
    scores = method_scores(d, verdicts)
    base = decisions_scores(d, d["gbm"] >= d["thr"])
    cascade_pair = decisions_scores(d, scores["cascade"] >= 0.5) if "cascade" in scores else None
    for row in curve:
        print(f"  LLM on {row['fraction']:6.2%} ({row['llm_pairs']:5d} pairs): F1 {row['f1']:.4f}")
    if everything:
        print(f"  LLM on everything (estimated): F1 {everything['f1']:.4f} {everything['f1_ci95']}")

    sizes = {}
    for f in sorted(LLM_DIR.glob("*_band.jsonl")):
        tag = f.stem
        rows = read_jsonl(f)
        sizes[tag] = {"band": band_comparison(d, rows), "cost": llm_cost(tag)}
    think = {}
    for f in sorted(LLM_DIR.glob("*think*.jsonl")):
        rows = read_jsonl(f)
        same = {p: band[p] for p in rows if p in band}
        think[f.stem] = {"thinking": band_comparison(d, rows), "no_thinking_same_pairs": band_comparison(d, same),
                         "cost": llm_cost(f.stem)}  # fmt: skip

    save_json(RESULTS / "cascade.json", {
        "meta": metadata(threshold=d["thr"], band=d["band"]),
        "candidates": N,
        "true_pairs": len(d["truth"]),
        "true_in_candidates": int(d["y"].sum()),
        "band_pairs": int(in_band.sum()),
        "gbm_only": base,
        "cascade_band": cascade_pair,
        "budget_curve": curve,
        "llm_everything": everything,
        "band_comparison": band_comparison(d, band),
        "clusters": clustering(d, scores),
        "recall_by_noise": recall_by_noise(d, scores),
        "hard_negatives": hard_negatives(d, scores),
        "examples": examples(d, scores, band, reasons),
        "prompt_dev": prompt_dev(),
        "gbm_band_accuracy": gbm_band_accuracy(d),
        "llm_runs": {t: llm_cost(t) for t in sorted({f.stem for f in LLM_DIR.glob("*.jsonl")})},
        "model_sizes": sizes,
        "thinking": think,
    })  # fmt: skip


if __name__ == "__main__":
    main()
