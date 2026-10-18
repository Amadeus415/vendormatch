"""Draw every chart in docs/ from the JSON in results/.

    uv run python -m bench.charts
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from .common import RESULTS, load_json  # noqa: E402

DOCS = RESULTS.parent / "docs"
ACCENT, INK, MUTED, GRID, OTHER = "#087F90", "#18202B", "#5B6B7D", "#D5DDE3", "#C2410C"
LIGHT = "#9AA8B6"  # de-emphasized series

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})  # fmt: skip

LABELS = {
    "exact": "Exact after normalize",
    "tfidf_cosine": "TF-IDF cosine",
    "rapidfuzz_ratio": "rapidfuzz ratio",
    "rapidfuzz_partial": "rapidfuzz partial",
    "rapidfuzz_token_sort": "rapidfuzz token_sort",
    "rapidfuzz_token_set": "rapidfuzz token_set",
    "jaro_winkler": "Jaro-Winkler",
    "logreg": "Logistic regression",
    "gbm": "Gradient boosting",
    "cascade": "GBM + LLM cascade",
}


def figure(title: str, subtitle: str, size=(8, 4.5), left: float = 0.10):
    """A figure with the finding as the title and the setup as a (wrapped) subtitle."""
    fig, ax = plt.subplots(figsize=size)
    lines = textwrap.wrap(subtitle, width=int(128 * size[0] / 8))
    inch = 1 / size[1]
    fig.text(0.04, 1 - 0.28 * inch, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.04, 1 - 0.62 * inch, "\n".join(lines), fontsize=9.5, color=MUTED, ha="left", va="top",
             linespacing=1.4)  # fmt: skip
    top = 1 - (0.95 + 0.2 * len(lines)) * inch
    fig.subplots_adjust(top=top, left=left, right=0.96, bottom=0.6 * inch)
    return fig, ax


def save(fig, name: str) -> None:
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote docs/{name}.png")


def setup(meta: dict) -> str:
    hw = meta["hardware"].split(",")[0]
    return f"{hw}, {meta['date']}"


# ---- 1. hero: cascade F1 vs share of pairs sent to the LLM ----------------------------------------------


def hero(c: dict, s: dict) -> None:
    curve = c["budget_curve"]
    x = np.array([r["fraction"] for r in curve])
    y = np.array([r["f1"] for r in curve])
    base = c["gbm_only"]["f1"]
    best_baseline = max(
        (m for k, m in s["methods"].items() if k not in ("gbm", "logreg")), key=lambda m: m["f1"]
    )
    band_frac = c["band_pairs"] / c["candidates"]
    casc = c["cascade_band"]["f1"] if c.get("cascade_band") else None
    everything = c.get("llm_everything")
    gain = (casc - base) if casc else 0
    title = f"Sending {band_frac:.1%} of pairs to the LLM lifts F1 from {base:.3f} to {casc:.3f}"
    sub = (f"Test split, {c['candidates']:,} candidate pairs. Qwen3-14B-4bit judges the pairs the GBM is least "
           f"sure about. {setup(c['meta'])}")  # fmt: skip
    fig, ax = figure(title, sub)
    xs = np.maximum(x, 1e-4)
    ax.plot(xs, y, color=ACCENT, lw=2, marker="o", ms=4, mec="white", mew=1, zorder=3)
    ax.axhline(base, color=LIGHT, lw=1)
    ax.text(1.05e-4, base - 0.004, f"GBM alone {base:.3f}", color=MUTED, fontsize=9, va="top")
    ax.axhline(best_baseline["f1"], color=OTHER, lw=1)
    ax.text(1.05e-4, best_baseline["f1"] + 0.003, f"Best string similarity ({LABELS[best_baseline['method']]}) "
            f"{best_baseline['f1']:.3f}", color=MUTED, fontsize=9, va="bottom")  # fmt: skip
    if casc:
        ax.scatter([band_frac], [casc], s=70, color=ACCENT, edgecolor="white", linewidth=2, zorder=4)
        ax.annotate(f"shipped band (GBM score 0.05-0.95)\n{band_frac:.1%} of pairs, F1 {casc:.3f} (+{gain:.3f})",
                    (band_frac, casc), xytext=(14, -38), textcoords="offset points", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})  # fmt: skip
    if everything:
        lo_ci, hi_ci = everything["f1_ci95"]
        ax.errorbar([1.0], [everything["f1"]], yerr=[[everything["f1"] - lo_ci], [hi_ci - everything["f1"]]],
                    fmt="o", color=INK, ms=6, mec="white", capsize=3, lw=1, zorder=4)  # fmt: skip
        ax.annotate(f"LLM on every pair\n(estimated) {everything['f1']:.3f}", (1.0, everything["f1"]),
                    xytext=(-12, 16), textcoords="offset points", ha="right", fontsize=9, color=INK)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(1e-4 * 0.8, 1.4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0%" if v <= 1.01e-4 else f"{v:.0%}" if v >= 0.01
                                               else f"{v:.1%}"))  # fmt: skip
    ax.set_xticks([1e-4, 0.001, 0.01, 0.1, 1.0])
    ax.set_xlabel("Share of candidate pairs sent to the LLM (log scale; most uncertain first)")
    ax.set_ylabel("Pair F1 on test")
    lo = min(best_baseline["f1"], base) - 0.03
    ax.set_ylim(lo, max(y.max(), casc or 0, everything["f1"] if everything else 0) + 0.02)
    save(fig, "hero_cascade_budget")


# ---- 2. method comparison -------------------------------------------------------------------------------


def methods_bar(s: dict, c: dict) -> None:
    order = ["exact", "tfidf_cosine", "rapidfuzz_ratio", "rapidfuzz_token_sort", "rapidfuzz_token_set",
             "jaro_winkler", "logreg", "gbm"]  # fmt: skip
    rows = [(LABELS[k], s["methods"][k]["precision"], s["methods"][k]["recall"], s["methods"][k]["f1"]) for k in order]
    if c.get("cascade_band"):
        cb = c["cascade_band"]
        rows.append((LABELS["cascade"], cb["precision"], cb["recall"], cb["f1"]))
    ceiling = c["true_in_candidates"] / c["true_pairs"]
    fig, ax = figure("Each stage adds accuracy; string similarity alone stalls below 0.90",
                     f"Pair F1 on test, thresholds tuned on dev. Recall counts all {c['true_pairs']:,} true pairs, and "
                     f"blocking caps it at {ceiling:.1%}. {setup(s['meta'])}", size=(8, 4.8), left=0.27)  # fmt: skip
    fig.subplots_adjust(right=0.78)
    y = np.arange(len(rows))
    for yi, (name, p, r, f) in zip(y, rows, strict=True):
        color = ACCENT if name == LABELS["cascade"] else INK if name in (LABELS["gbm"], LABELS["logreg"]) else LIGHT
        ax.scatter([f], [yi], s=70, color=color, edgecolor="white", linewidth=2, zorder=3)
        ax.text(f + 0.012, yi, f"{f:.3f}", va="center", fontsize=9.5, color=INK)
        ax.text(1.03, yi, f"P {p:.3f}   R {r:.3f}", va="center", fontsize=8.5, color=MUTED,
                transform=ax.get_yaxis_transform())  # fmt: skip
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.6, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 (axis starts at 0.6)")
    save(fig, "methods_f1")


# ---- 3. PR curves ----------------------------------------------------------------------------------------


def pr_curves(s: dict) -> None:
    fig, ax = figure("The learned scorer beats every single similarity at every threshold",
                     f"Precision-recall on test candidate pairs; recall is out of all true pairs, so no curve reaches "
                     f"1.0. {setup(s['meta'])}")  # fmt: skip
    show = [("gbm", ACCENT, 2.2), ("logreg", INK, 1.4), ("jaro_winkler", OTHER, 1.4),
            ("rapidfuzz_token_set", LIGHT, 1.4), ("tfidf_cosine", LIGHT, 1.0)]  # fmt: skip
    handles = []
    for key, color, lw in show:
        m = s["methods"][key]
        r, p = np.array(m["pr_curve"]["recall"]), np.array(m["pr_curve"]["precision"])
        (h,) = ax.plot(r, p, color=color, lw=lw, ls=(0, (4, 2)) if key == "tfidf_cosine" else "-")
        h.set_label(f"{LABELS[key]}  AP {m['average_precision']:.3f}")
        handles.append(h)
    ex = s["methods"]["exact"]
    h = ax.scatter([ex["recall"]], [ex["precision"]], color=MUTED, s=26, zorder=3,
                   label=f"{LABELS['exact']}  (one point)")  # fmt: skip
    handles.append(h)
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9, handlelength=2.2)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.6, 1.005)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    save(fig, "pr_curves")


# ---- 4. blocking -----------------------------------------------------------------------------------------


def blocking(b: dict) -> None:
    test = b["test"]
    n = test["records"]
    total = n * (n - 1) / 2
    chosen = _row(test, "tfidf_knn+acronym", b["chosen_k"])
    fig, ax = figure(f"Blocking keeps {chosen['recall']:.1%} of true pairs while scoring "
                     f"{chosen['candidates'] / total:.2%} of all pairs",
                     f"Test split, {n:,} records, {total:,.0f} possible pairs, {test['true_pairs']:,} true pairs. "
                     f"Circle = k={b['chosen_k']}, picked on dev. {setup(b['meta'])}")  # fmt: skip
    series = [("tfidf_knn+acronym", ACCENT, "k", "k={}", {1, 3, 5, 10, 50}),
              ("token", OTHER, "max_df", "df≤{}", {5, 10, 50, 250})]  # fmt: skip
    for method, color, key, fmt, shown in series:
        rows = [r for r in test["rows"] if r["method"] == method]
        ax.plot([r["candidates"] for r in rows], [r["recall"] for r in rows], color=color, lw=2, marker="o", ms=4,
                mec="white", mew=1)  # fmt: skip
        for r in rows:
            if r[key] in shown:
                ax.annotate(fmt.format(r[key]), (r["candidates"], r["recall"]), xytext=(5, -12),
                            textcoords="offset points", fontsize=8, color=MUTED)  # fmt: skip
    knn = [r for r in test["rows"] if r["method"] == "tfidf_knn+acronym"]
    tok = [r for r in test["rows"] if r["method"] == "token"]
    ax.annotate("TF-IDF character 3-grams,\nk nearest neighbours + acronym", (knn[-1]["candidates"], knn[-1]["recall"]),
                xytext=(0, 14), textcoords="offset points", ha="right", fontsize=9, color=INK)  # fmt: skip
    ax.annotate("records sharing a rare word", (tok[-1]["candidates"], tok[-1]["recall"]), xytext=(0, -28),
                textcoords="offset points", ha="right", fontsize=9, color=INK)  # fmt: skip
    ax.scatter([chosen["candidates"]], [chosen["recall"]], s=110, facecolor="none", edgecolor=INK, lw=1.2, zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.set_xlabel("Candidate pairs to score (log scale)")
    ax.set_ylabel("True pairs kept")
    save(fig, "blocking_recall")


def _row(split: dict, method: str, k: int) -> dict:
    return next(r for r in split["rows"] if r["method"] == method and r.get("k") == k)


# ---- 5. threshold sweep ----------------------------------------------------------------------------------


def threshold_sweep(s: dict) -> None:
    m = s["methods"]["gbm"]
    sw = m["sweep_test"]
    t = [r["threshold"] for r in sw]
    fig, ax = figure("F1 is flat between 0.2 and 0.6, so the dev-chosen threshold transfers",
                     f"Gradient boosting on test, threshold swept. Dashed line = threshold picked on dev "
                     f"({m['threshold']:.2f}). {setup(s['meta'])}")  # fmt: skip
    ax.plot(t, [r["f1"] for r in sw], color=ACCENT, lw=2)
    ax.plot(t, [r["precision"] for r in sw], color=INK, lw=1.2)
    ax.plot(t, [r["recall"] for r in sw], color=OTHER, lw=1.2)
    ax.text(t[-1] + 0.01, sw[-1]["f1"], "F1", color=INK, fontsize=9, va="center")
    ax.text(t[-1] + 0.01, sw[-1]["precision"], "precision", color=INK, fontsize=9, va="center")
    ax.text(t[-1] + 0.01, sw[-1]["recall"], "recall", color=INK, fontsize=9, va="center")
    ax.axvline(m["threshold"], color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Match threshold on GBM probability")
    ax.set_ylabel("Score on test")
    save(fig, "threshold_sweep")


# ---- 6. recall by corruption -----------------------------------------------------------------------------


NOISE_LABELS = {
    "suffix": "legal suffix changed", "punctuation": "punctuation / &", "abbreviation": "abbreviation (Intl, Mfg)",
    "typo": "typo", "card_descriptor": "card descriptor (caps, 22 chars)",
    "processor_prefix": "processor prefix (SQ *)", "store_number": "store number", "truncated": "truncated",
    "charge_city": "card city is the charge city", "attention_line": "attention line",
    "remit_address": "remit-to lockbox address", "dba": "dba name",
    "alias": "brand / alias (Wikidata)", "acronym": "acronym (IBM)", "former_name": "former name (SEC)",
}  # fmt: skip


def recall_by_noise(c: dict) -> None:
    rb = c["recall_by_noise"]
    tags = sorted(rb, key=lambda t: rb[t].get("cascade", rb[t]["gbm"]))
    fig, ax = figure("Former names and aliases are where every method loses recall",
                     f"Share of true test pairs found, by what was done to either record. Dot = method, bar end = "
                     f"blocking ceiling. {setup(c['meta'])}", size=(8, 6.2))  # fmt: skip
    fig.subplots_adjust(left=0.33, top=0.86, bottom=0.09)
    y = np.arange(len(tags))
    ax.barh(y, [rb[t]["blocking"] for t in tags], height=0.5, color="#EEF2F5")
    series = [("rapidfuzz_token_set", OTHER, "token_set"), ("gbm", INK, "GBM"), ("cascade", ACCENT, "GBM + LLM")]
    for key, color, _ in series:
        if key in rb[tags[0]]:
            ax.scatter([rb[t][key] for t in tags], y, color=color, s=28, zorder=3, edgecolor="white", linewidth=1)
    top = tags[-1]
    for key, _, label in series:
        if key in rb[top]:
            ax.annotate(label, (rb[top][key], y[-1]), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)  # fmt: skip
    ax.set_yticks(y, [f"{NOISE_LABELS.get(t, t)}  ({rb[t]['true_pairs']:,})" for t in tags])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Recall on true pairs touched by this corruption (pair count in brackets)")
    save(fig, "recall_by_corruption")


# ---- 7. inside the band: GBM vs LLM, and model size -------------------------------------------------------


def band_models(c: dict) -> None:
    sizes = {k: v for k, v in c.get("model_sizes", {}).items() if v.get("band") and v.get("cost")}
    if not sizes:
        return
    fig, ax = figure("Inside the uncertain band, bigger judges are slower and more accurate",
                     f"F1 on the {c['band_pairs']:,} test pairs with GBM score 0.05-0.95, thinking off, one pair at a "
                     f"time with a reason. {setup(c['meta'])}")  # fmt: skip
    gbm_f1 = next(iter(sizes.values()))["band"]["gbm"]["f1"]
    ax.axhline(gbm_f1, color=OTHER, lw=1)
    ax.text(0.02, gbm_f1 + 0.005, f"GBM alone on these pairs: F1 {gbm_f1:.3f}", color=MUTED, fontsize=9,
            transform=ax.get_yaxis_transform())  # fmt: skip
    for tag, v in sorted(sizes.items(), key=lambda kv: kv[1]["cost"]["mean_seconds_per_pair"]):
        x, y = v["cost"]["mean_seconds_per_pair"], v["band"]["llm"]["f1"]
        ax.scatter([x], [y], s=60, color=ACCENT if "14b" in tag else INK, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{tag.split('_')[0]}  F1 {y:.3f}", (x, y), xytext=(8, -3), textcoords="offset points",
                    fontsize=9, color=INK)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Seconds per pair (log scale)")
    ax.set_ylabel("F1 on band pairs")
    save(fig, "llm_model_size")


# ---- 8. scaling ------------------------------------------------------------------------------------------


def scaling(sc: dict) -> None:
    rows = sc["rows"]
    n = np.array([r["records"] for r in rows])
    title = f"{rows[-1]['records'] // 1000}k records dedupe in {rows[-1]['total_seconds']:.0f} s on a laptop, no LLM"
    sub = f"Wall time per stage vs number of records (log-log), k={sc['meta']['k']}. {setup(sc['meta'])}"
    fig, ax = figure(title, sub)
    stages = [("features", ACCENT), ("block", INK), ("cluster", OTHER), ("normalize", LIGHT), ("score", LIGHT)]
    for stage, color in stages:
        t = np.array([r["seconds"][stage] for r in rows])
        ax.plot(n, t, color=color, lw=1.8, marker="o", ms=3.5, mec="white", mew=0.8)
        ax.text(n[-1] * 1.08, t[-1], stage, fontsize=9, color=INK, va="center")
    total = np.array([r["total_seconds"] for r in rows])
    ax.plot(n, total, color=INK, lw=1, ls=(0, (1, 2)))
    ax.text(n[-1] * 1.08, total[-1], f"total ({rows[-1]['records_per_second']:,.0f} rec/s)", fontsize=9,
            color=INK, va="center")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n[0] * 0.8, n[-1] * 3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Records")
    ax.set_ylabel("Seconds")
    save(fig, "scaling")


# ---- 9. clustering -----------------------------------------------------------------------------------------


def clusters(c: dict) -> None:
    cl = c["clusters"]
    order = [k for k in ["exact", "tfidf_cosine", "rapidfuzz_token_set", "jaro_winkler", "gbm", "cascade"] if k in cl]
    sub = f"Cluster-level pairwise F1 on test, same pair scores clustered two ways. {setup(c['meta'])}"
    fig, ax = figure("Average linkage repairs the over-merging union-find causes", sub)
    fig.subplots_adjust(left=0.26)
    y = np.arange(len(order))
    uf = [cl[k]["union_find"]["pairwise_f1"] for k in order]
    al = [cl[k]["average_link"]["pairwise_f1"] for k in order]
    for yi, a, b in zip(y, uf, al, strict=True):
        ax.plot([a, b], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.scatter(uf, y, color=OTHER, s=40, zorder=3, edgecolor="white", linewidth=1.5)
    ax.scatter(al, y, color=ACCENT, s=40, zorder=3, edgecolor="white", linewidth=1.5)
    ax.annotate("union-find", (uf[0], y[0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
    ax.annotate("average link", (al[0], y[0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
    for yi, a, b in zip(y, uf, al, strict=True):
        ax.text(max(a, b) + 0.01, yi, f"{b:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [LABELS[k] for k in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pairwise F1 of the final clusters")
    save(fig, "clustering")


def main() -> None:
    s = load_json(RESULTS / "scorers.json")
    b = load_json(RESULTS / "blocking.json")
    blocking(b)
    pr_curves(s)
    threshold_sweep(s)
    cpath = RESULTS / "cascade.json"
    if cpath.exists():
        c = load_json(cpath)
        methods_bar(s, c)
        if c["budget_curve"]:
            hero(c, s)
        recall_by_noise(c)
        band_models(c)
        clusters(c)
    spath = RESULTS / "scaling.json"
    if spath.exists():
        scaling(load_json(spath))


if __name__ == "__main__":
    main()
