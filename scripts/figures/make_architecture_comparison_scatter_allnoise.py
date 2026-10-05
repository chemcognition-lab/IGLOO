#!/usr/bin/env python3
"""
Same as make_architecture_comparison_scatter.py, but aggregating across ALL
noise levels (and all landscapes and seeds) instead of each architecture's
own best-performance noise level. This is the "full-sweep average" view,
in contrast to the project's established best-noise-operating-point
convention used in the other script -- useful as a direct comparison of
the two methodologies, not a replacement.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import topsis_score, BACKBONE_METRICS, PROTEINS  # noqa: E402

METHOD_DISPLAY = {
    "IG": "Pool First (baseline)",
    "PreMLP": "Residual Pre-MLP",
    "MeanPool": "Per-residue + Mean Pool",
    "AttnPool": "Per-residue + Attn Pool",
    "OneHot": "One-Hot (baseline)",
}
METHOD_COLORS = {
    "IG": "#E69F00", "PreMLP": "#009E73", "MeanPool": "#0072B2",
    "AttnPool": "#CC79A7", "OneHot": "#999999",
}
METHOD_MARKERS = {"IG": "o", "PreMLP": "s", "MeanPool": "D", "AttnPool": "^", "OneHot": "X"}


def main():
    df_all = pd.read_csv("results/synthetic/metric_suite/metric_suite.csv")
    df = df_all[df_all.backbone == "esmc_300m"].copy()
    composite_metrics = [m for m in BACKBONE_METRICS if m in df.columns]
    df["topsis_score"] = topsis_score(df, composite_metrics)
    sweep = df[df.kind == "sweep"]

    methods = list(METHOD_DISPLAY.keys())
    proteins = [p for p in PROTEINS if p in sweep.protein.unique()]

    fig, axes = plt.subplots(1, len(proteins), figsize=(6.5 * len(proteins), 8.0))
    if len(proteins) == 1:
        axes = [axes]

    for ax, protein in zip(axes, proteins):
        p_df = sweep[sweep.protein == protein]

        for method in methods:
            m_df = p_df[p_df.method == method]
            if method == "IG":
                m_df = m_df[m_df.seed != 0]
            m_df = m_df.dropna(subset=["model_spearman", "topsis_score"])
            if m_df.empty:
                continue

            x_mean, x_std = m_df.model_spearman.mean(), m_df.model_spearman.std()
            y_mean, y_std = m_df.topsis_score.mean(), m_df.topsis_score.std()

            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std,
                        fmt=marker, color=color, ecolor=color, elinewidth=1.3,
                        capsize=4, markersize=11, markeredgecolor="black",
                        markeredgewidth=0.8, label=METHOD_DISPLAY[method], zorder=3)

        ax.set_title(f"{protein}", fontweight="bold", fontsize=16, color="black")
        ax.set_xlabel("Model Test Spearman ρ", fontsize=13, color="black")
        ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=13, color="black")
        ax.tick_params(axis="both", labelsize=11, labelcolor="black", color="black")
        ax.set_ylim(-0.02, 1.02)
        legend = ax.legend(loc="lower right", fontsize=11, title="Architecture", title_fontsize=11)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.text(0.03, 0.97, f"{protein}, ESM-C 300M\nerror bars = std across ALL noise levels,\nlandscapes, and seeds (full-sweep average)",
                transform=ax.transAxes, fontsize=10, va="top", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.2)

    fig.suptitle("Predictive Accuracy vs Attribution Quality — Architecture Comparison (full noise sweep)",
                 fontweight="bold", fontsize=17, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/synthetic/metric_suite/architecture_comparison_scatter_allnoise"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
