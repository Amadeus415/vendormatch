"""Stage 4 benchmark: run the local LLM judge over test candidate pairs.

Resumable and chunked, so long runs can share the GPU: each call judges up to --limit new pairs and
appends to results/llm/<tag>.jsonl.

    uv run python -m bench.run_llm --tag qwen3-14b_all --pairs all --no-reason
    uv run python -m bench.run_llm --tag qwen3-14b_band --pairs band
    uv run python -m bench.run_llm --tag qwen3-14b_think256 --pairs band --think 256 --sample 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time

from vendormatch.llm import DEFAULT_MODEL, DEFAULT_PROMPT, PROMPTS, LLMJudge

from .common import RESULTS, load_json, load_split, metadata, save_json

LLM_DIR = RESULTS / "llm"


def uncertainty(p: float, threshold: float) -> float:
    """Distance from the decision threshold in log-odds. Small = the model is unsure."""
    eps = 1e-6
    logit = lambda x: math.log(max(x, eps) / max(1 - x, eps))  # noqa: E731
    return abs(logit(p) - logit(threshold))


def load_pairs(which: str, split: str = "test") -> list[dict]:
    """'all': every candidate, most uncertain first, so a partial run already covers the cascade.
    'band': only pairs in the model's uncertain band, in a fixed random order.
    On dev the GBM scores are out-of-fold, so the dev band looks like the test band."""
    rows = list(csv.DictReader((RESULTS / f"{split}_pairs.csv").open()))
    meta = load_json(RESULTS / "scorers.json")
    lo, hi = meta["meta"]["band"]
    thr = meta["methods"]["gbm"]["threshold"]
    out = [{"a": int(r["a"]), "b": int(r["b"]), "label": int(r["label"]), "gbm": float(r["gbm"])} for r in rows]
    if which == "band":
        out = [r for r in out if lo <= r["gbm"] <= hi]
        random.Random(0).shuffle(out)
    else:
        out.sort(key=lambda r: (uncertainty(r["gbm"], thr), r["a"], r["b"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    ap.add_argument("--model-id", default=None, help="name to record when --model is a local path")
    ap.add_argument("--split", choices=["test", "dev"], default="test", help="dev is for prompt development")
    ap.add_argument("--pairs", choices=["all", "band", "tail"], default="band")
    ap.add_argument("--head", type=int, default=0, help="with --pairs tail: skip the N most uncertain pairs")
    ap.add_argument("--sample", type=int, default=0, help="only the first N pairs of the fixed order")
    ap.add_argument("--limit", type=int, default=0, help="judge at most N new pairs this call")
    ap.add_argument("--no-reason", action="store_true")
    ap.add_argument("--think", type=int, default=0, help="thinking budget in tokens (0 = thinking off)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT)
    ap.add_argument("--reuse", default=None, help="tag of an earlier run with the same model; skip its pairs")
    args = ap.parse_args()

    records = load_split(args.split)
    if args.pairs == "tail":  # a uniform random sample of the pairs after the exhaustive head
        pairs = load_pairs("all", args.split)[args.head :]
        random.Random(1).shuffle(pairs)
    else:
        pairs = load_pairs(args.pairs, args.split)
    if args.sample:
        pairs = pairs[: args.sample]
    out_path = LLM_DIR / f"{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(r["a"], r["b"]) for r in map(json.loads, out_path.open())}
    if args.reuse:  # the decision token is the same with or without a reason, so these are not judged twice
        done |= {(r["a"], r["b"]) for r in map(json.loads, (LLM_DIR / f"{args.reuse}.jsonl").open())}
    todo = [p for p in pairs if (p["a"], p["b"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.tag}: {len(pairs)} pairs, {len(done)} done, judging {len(todo)} now", flush=True)
    if not todo:
        return

    t_load = time.perf_counter()
    judge = LLMJudge(args.model, reason=not args.no_reason, thinking_budget=args.think, prompt=args.prompt)
    judge.warm()
    t_load = time.perf_counter() - t_load
    t0 = time.perf_counter()
    with out_path.open("a") as f:
        for n, p in enumerate(todo, 1):
            j = judge.judge(records[p["a"]], records[p["b"]])
            f.write(json.dumps({**p, **j.to_dict()}) + "\n")
            if n % 100 == 0:
                f.flush()
                print(f"  {n}/{len(todo)}  {n / (time.perf_counter() - t0):.2f} pairs/s", flush=True)
    elapsed = time.perf_counter() - t0
    meta_path = LLM_DIR / f"{args.tag}.meta.json"
    runs = load_json(meta_path)["runs"] if meta_path.exists() else []
    runs.append({"pairs": len(todo), "seconds": elapsed, "load_seconds": t_load, "started_after": len(done),
                 **metadata()})  # fmt: skip
    save_json(meta_path, {"model": args.model_id or args.model, "split": args.split, "pairs": args.pairs,
                          "sample": args.sample,
                          "reason": not args.no_reason, "thinking_budget": args.think, "prompt": args.prompt,
                          "total_pairs": len(pairs), "runs": runs})  # fmt: skip
    print(f"done: {len(todo)} pairs in {elapsed:.0f}s ({len(todo) / elapsed:.2f} pairs/s)")


if __name__ == "__main__":
    main()
