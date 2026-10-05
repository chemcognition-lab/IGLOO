#!/usr/bin/env python3
"""
TOPSIS vs model performance, one plot per protein with all 3 landscapes
overlaid, each landscape's Pearson r (model_spearman vs topsis_score)
annotated directly on the plot. Generates one such figure per architecture:
IG ("Pool First"), MeanPool, AttnPool.

Uses the exact same topsis_score()/kind=="sweep" methodology as
metric_suite.py's plot_metric_vs_perf_band() and perf_attribution_correlation_table().
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import topsis_score, BACKBONE_METRICS, PROTEINS, LANDSCAPES  # noqa: E402

LANDSCAPE_COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}
LANDSCAPE_MARKERS = {"additive": "o", "epistatic": "s", "nonlinear": "D"}

# method -> (display name, expected consistent sample size per noise level,
# output filename slug). IG uses only the 5-seed seedsweep ensemble (seed=0
# is a different default-sweep run from a separate experiment, excluded for
# a clean single-source comparison); MeanPool/AttnPool come from their own
# single 3-seed directory tree (n=3, no orphans).
METHODS = {
    "IG": ("Pool First (IG)", 5, "poolfirst_topsis_by_landscape"),
    "MeanPool": ("MeanPool", 3, "meanpool_topsis_by_landscape"),
    "AttnPool": ("AttnPool", 3, "attnpool_topsis_by_landscape"),
}


def make_figure(df, method, display_name, expected_n, out_slug):
    sweep = df[(df.kind == "sweep") & (df.method == method)]
    if method == "IG":
        sweep = sweep[sweep.seed != 0]

    proteins = [p for p in PROTEINS if p in sweep.protein.unique()]
    fig, axes = plt.subplots(1, len(proteins), figsize=(6.5 * len(proteins), 8.0))
    if len(proteins) == 1:
        axes = [axes]

    for ax, protein in zip(axes, proteins):
        p_df = sweep[sweep.protein == protein]
        annot_lines = []
        for landscape in LANDSCAPES:
            g = p_df[p_df.landscape == landscape]
            if g.empty:
                continue
            agg = g.groupby("noise").agg(
                rho_mean=("model_spearman", "mean"), rho_std=("model_spearman", "std"),
                topsis_mean=("topsis_score", "mean"), topsis_std=("topsis_score", "std"),
                n=("topsis_score", "count"),
            ).reset_index().sort_values("rho_mean")
            # keep only noise levels with the expected consistent sample size
            # (drops any orphaned/non-standard-grid single-run points)
            agg = agg[agg.n == expected_n]

            color = LANDSCAPE_COLORS[landscape]
            marker = LANDSCAPE_MARKERS[landscape]
            ax.plot(agg.rho_mean, agg.topsis_mean, f"{marker}-", color=color,
                    markersize=6, linewidth=1.5, alpha=0.95, label=landscape.capitalize())
            ax.fill_between(agg.rho_mean, agg.topsis_mean - agg.topsis_std,
                             agg.topsis_mean + agg.topsis_std, color=color, alpha=0.2, linewidth=0)

            s = g.dropna(subset=["model_spearman", "topsis_score"])
            r, pval = pearsonr(s.model_spearman, s.topsis_score)
            annot_lines.append(f"{landscape.capitalize()}: Pearson r = {r:.3f} (p={pval:.1e})")

        ax.set_title(f"{protein} — {display_name}", fontweight="bold", fontsize=14, color="black")
        ax.set_xlabel("Model Test Spearman ρ", fontsize=12, color="black")
        ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=12, color="black")
        ax.tick_params(axis="both", labelsize=10, labelcolor="black", color="black")
        ax.set_ylim(-0.02, 1.02)
        legend = ax.legend(loc="lower right", fontsize=10, title="Landscape", title_fontsize=10)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.text(0.03, 0.97, "\n".join(annot_lines), transform=ax.transAxes,
                fontsize=9, va="top", ha="left", color="black",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.9))
        ax.text(0.03, 0.03, f"{protein}, ESM-C 300M, {expected_n} seeds", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.1)

    fig.suptitle(f"TOPSIS Score vs Model Performance — {display_name} architecture",
                 fontweight="bold", fontsize=15, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = f"results/synthetic/metric_suite/{out_slug}"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png / .pdf")


def main():
    df_all = pd.read_csv("results/synthetic/metric_suite/metric_suite.csv")
    df = df_all[df_all.backbone == "esmc_300m"].copy()
    composite_metrics = [m for m in BACKBONE_METRICS if m in df.columns]
    df["topsis_score"] = topsis_score(df, composite_metrics)

    for method, (display_name, expected_n, out_slug) in METHODS.items():
        make_figure(df, method, display_name, expected_n, out_slug)


if __name__ == "__main__":
    main()
