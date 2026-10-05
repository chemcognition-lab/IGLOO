#!/usr/bin/env python3
"""
Predictive accuracy (model test Spearman rho, x) vs attribution quality
(TOPSIS, y) scatter with error bars on both axes, one point per architecture:
IG (Pool First baseline), PreMLP (per-residue MLP), MeanPool (per-residue +
mean pool), AttnPool (per-residue + attention pool), OneHot (one-hot input
baseline). One subplot per protein.

Methodology matches the project's established best-noise-operating-point
convention (and the Figure-1 Panel D backbone scatter): for each
(method, protein, landscape), pick the noise level with the highest mean
model_spearman across seeds, read model_spearman/topsis_score there, then
aggregate across the 3 landscapes (mean +/- std) to get one point per
architecture. IG uses only the 5-seed seedsweep ensemble (seed=0, the
separate default-sweep run, is excluded for consistency with the other
poolfirst_* figures).

Same visual conventions as make_poolfirst_topsis_figure.py.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import topsis_score, BACKBONE_METRICS, PROTEINS, LANDSCAPES  # noqa: E402

METHOD_DISPLAY = {
    "IG": "Pool First (baseline)",
    "PreMLP": "Residual Pre-MLP",
    "MeanPool": "Per-residue + Mean Pool",
    "AttnPool": "Per-residue + Attn Pool",
    "OneHot": "One-Hot (baseline)",
}
METHOD_COLORS = {
    "IG": "#E69F00",
    "PreMLP": "#009E73",
    "MeanPool": "#0072B2",
    "AttnPool": "#CC79A7",
    "OneHot": "#999999",
}
METHOD_MARKERS = {
    "IG": "o", "PreMLP": "s", "MeanPool": "D", "AttnPool": "^", "OneHot": "X",
}


def best_noise_point(g):
    """g: rows for one (method, protein, landscape), kind=='sweep'. Returns
    (model_spearman_mean_at_best_noise, topsis_mean_at_best_noise) using the
    noise level with the highest mean model_spearman across seeds."""
    by_noise = g.groupby("noise")["model_spearman"].mean()
    if by_noise.empty:
        return None
    best_noise = by_noise.idxmax()
    sub = g[g.noise == best_noise]
    return sub["model_spearman"].mean(), sub["topsis_score"].mean()


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
            if m_df.empty:
                continue

            land_points = []
            for landscape in LANDSCAPES:
                g = m_df[m_df.landscape == landscape]
                if g.empty:
                    continue
                pt = best_noise_point(g)
                if pt is not None:
                    land_points.append(pt)
            if not land_points:
                continue

            xs = np.array([p[0] for p in land_points])
            ys = np.array([p[1] for p in land_points])
            x_mean, x_std = xs.mean(), xs.std()
            y_mean, y_std = ys.mean(), ys.std()

            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std,
                        fmt=marker, color=color, ecolor=color, elinewidth=1.3,
                        capsize=4, markersize=11, markeredgecolor="black",
                        markeredgewidth=0.8, label=METHOD_DISPLAY[method], zorder=3)

        ax.set_title(f"{protein}", fontweight="bold", fontsize=14, color="black")
        ax.set_xlabel("Model Test Spearman ρ", fontsize=12, color="black")
        ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=12, color="black")
        ax.tick_params(axis="both", labelsize=10, labelcolor="black", color="black")
        ax.set_ylim(-0.02, 1.02)
        legend = ax.legend(loc="lower right", fontsize=10, title="Architecture", title_fontsize=10)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.text(0.03, 0.97, f"{protein}, ESM-C 300M\nerror bars = std across 3 landscapes\n(at each architecture's own best-performance noise level)",
                transform=ax.transAxes, fontsize=9, va="top", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.1)

    fig.suptitle("Predictive Accuracy vs Attribution Quality — Architecture Comparison",
                 fontweight="bold", fontsize=15, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/synthetic/metric_suite/architecture_comparison_scatter"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
