#!/usr/bin/env python3
"""
Focused version of the metric_suite figures: just IG, IG_premlp, perres_mean,
perres_attn, ISM (the "hero" comparison — baseline, the small architecture
fix, the two winning per-residue architectures, and the gradient-free
cross-check) with error bands showing the spread across seeds, so overlap
between methods is easy to read.

For each (protein, landscape, method), points are grouped by noise level and
summarized as mean +/- std across seeds; the per-noise-level means are
connected by a line with a shaded +/-1 std band. Raw per-seed points are also
drawn lightly underneath so the actual spread (not just the summary) is
visible.

Reuses the already-computed results/synthetic/metric_suite/metric_suite.csv
(run scripts/metric_suite.py first / after the rebuild) and the random-null
baseline helper from that script.

Usage:
    python scripts/metric_suite_focused.py [--root .../results/synthetic] [--out .../metric_suite]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.metric_suite import _null_for, _METRICS, _SIGNED_METRICS, PROTEINS, LANDSCAPES

METHODS = ["IG", "IG_premlp", "perres_mean", "perres_attn", "ISM"]
METHOD_LABEL = {
    "IG": "IG (baseline, esm_attention)",
    "IG_premlp": "IG + premlp",
    "perres_mean": "per-residue MLP, mean pool",
    "perres_attn": "per-residue MLP, attn pool",
    "ISM": "ISM (reference, gradient-free)",
}
# color = method identity; consistent with the main metric_suite.py palette
COLOR = {
    "IG": "#d62728", "IG_premlp": "#2ca02c",
    "perres_mean": "#17becf", "perres_attn": "#ff7f0e", "ISM": "#1f77b4",
}
MARKER = {"IG": "o", "IG_premlp": "D", "perres_mean": "h", "perres_attn": "H", "ISM": "s"}

ALL_METRICS = list(_METRICS) + list(_SIGNED_METRICS)


def main():
    project_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=project_root / "results" / "synthetic")
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "metric_suite")
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    help="subset of methods to overlay (default: IG IG_premlp perres_mean perres_attn ISM)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.root / "metric_suite" / "metric_suite.csv"
    if not csv_path.exists():
        print(f"missing {csv_path} -- run scripts/metric_suite.py first"); return
    df = pd.read_csv(csv_path)
    sweep = df[df.kind == "sweep"]

    methods = [m for m in args.methods if m in sweep.method.unique()]
    missing = [m for m in args.methods if m not in methods]
    if missing:
        print(f"note: no sweep data yet for {missing} (still pending / pilot not run) -- skipping")

    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not sweep[(sweep.protein == p) & (sweep.landscape == l)].empty]

    for m in ALL_METRICS:
        ncol = 3
        nrow = int(np.ceil(len(combos) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.6 * nrow),
                                 squeeze=False, layout="constrained")
        for ax, (p, l) in zip(axes.flat, combos):
            g_pl = sweep[(sweep.protein == p) & (sweep.landscape == l)]

            rm = _null_for(p, m)
            if rm is not None:
                lab = "chance" if (m.startswith("skill_") or m == "sign_acc") else "random ranking"
                ax.axhline(rm, color="#888", ls=":", lw=1.2, zorder=0)
                ax.text(0.99, rm, f" {lab}", color="#666", fontsize=7,
                        ha="right", va="bottom", transform=ax.get_yaxis_transform())

            for method in methods:
                sub = g_pl[g_pl.method == method].dropna(subset=["model_spearman", m])
                if sub.empty:
                    continue
                c = COLOR[method]

                # raw per-seed points, lightly, underneath -- the actual spread
                ax.scatter(sub["model_spearman"], sub[m], color=c, s=16, alpha=0.30,
                           edgecolors="none", zorder=2)

                # per-noise-level mean +/- std, connected, shaded band
                agg = (sub.groupby("noise")
                          .agg(mx=("model_spearman", "mean"), my=(m, "mean"), sy=(m, "std"),
                               n=(m, "size"))
                          .reset_index().sort_values("mx"))
                agg["sy"] = agg["sy"].fillna(0.0)
                if len(agg) >= 2:
                    ax.plot(agg["mx"], agg["my"], color=c, lw=2.0, zorder=4,
                            marker=MARKER[method], ms=6, mec="black", mew=0.4,
                            label=f"{METHOD_LABEL[method]} (n={int(agg['n'].sum())})")
                    ax.fill_between(agg["mx"], agg["my"] - agg["sy"], agg["my"] + agg["sy"],
                                    color=c, alpha=0.18, zorder=1, linewidth=0)
                else:
                    # only one noise level with data -- still show mean +/- std as an errorbar
                    ax.errorbar(agg["mx"], agg["my"], yerr=agg["sy"], fmt=MARKER[method],
                                color=c, ms=7, mec="black", mew=0.4, capsize=3, zorder=4,
                                label=f"{METHOD_LABEL[method]} (n={int(agg['n'].sum())})")

            ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
            ax.set_xlabel("model test Spearman ρ (predicted vs true fitness)")
            ax.set_ylabel(m)
            ax.legend(fontsize=6.5, loc="best")
        for ax in axes.flat[len(combos):]:
            ax.set_visible(False)
        fig.suptitle(f"{m} vs model performance — focused comparison\n"
                     f"line = mean across seeds per noise level, shaded band = ±1 std, "
                     f"dots = individual seeds",
                     fontsize=11, fontweight="bold")
        for ext in ("png", "svg"):
            fig.savefig(args.out / f"metric_{m}_focused.{ext}", dpi=150)
        plt.close(fig)

    print(f"saved focused plots to {args.out} (metric_<name>_focused.png)")


if __name__ == "__main__":
    main()
