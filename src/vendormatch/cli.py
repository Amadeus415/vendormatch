"""Command line: `vendormatch dedupe input.csv --out clusters.csv [--llm]`."""

from __future__ import annotations

import argparse
import csv
import sys

from .llm import DEFAULT_MODEL


def _load_llm(args):
    if not args.llm:
        return None
    from .llm import LLMJudge

    print(f"loading {args.llm_model} ...", file=sys.stderr)
    judge = LLMJudge(args.llm_model)
    judge.warm()
    return judge


def cmd_dedupe(args) -> None:
    from .pipeline import dedupe

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("input has no rows")
    if args.name_col not in rows[0]:
        sys.exit(f"no '{args.name_col}' column; pass --name-col")
    records = [{**r, "name": r[args.name_col]} for r in rows]
    result = dedupe(records, llm=_load_llm(args), k=args.k)

    sizes: dict[int, int] = {}
    for c in result.clusters:
        sizes[int(c)] = sizes.get(int(c), 0) + 1
    fields = list(rows[0]) + ["cluster_id", "cluster_size", "golden_name"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, c in zip(rows, result.clusters, strict=True):
            w.writerow({**r, "cluster_id": int(c), "cluster_size": sizes[int(c)], "golden_name": result.golden[int(c)]})
    n_llm = result.decided_by.count("llm")
    secs = sum(result.seconds.values())
    print(
        f"{len(rows)} records -> {len(sizes)} clusters ({sum(1 for s in sizes.values() if s > 1)} with duplicates); "
        f"{len(result.pairs)} candidate pairs, {n_llm} sent to the LLM; {secs:.1f}s. Wrote {args.out}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vendormatch", description="Find duplicate vendor/company records.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dedupe", help="cluster the records in a CSV")
    d.add_argument("input")
    d.add_argument("--out", required=True)
    d.add_argument("--name-col", default="name")
    d.add_argument("--k", type=int, default=10, help="neighbours per record in blocking")
    d.set_defaults(func=cmd_dedupe)

    d.add_argument("--llm", action="store_true", help="send unsure pairs to a local MLX model (Apple Silicon)")
    d.add_argument("--llm-model", default=DEFAULT_MODEL, help="HF repo id or local path of an MLX model")

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
