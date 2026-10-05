#!/usr/bin/env python3
"""
Heatmap of the 5 shortlisted metrics (skill_spearman, skill_auroc,
skill_precision_at_k, skill_recall_topcat, skill_sign_acc) across the 5
architectures (IG/Pool First, PreMLP, MeanPool, AttnPool, OneHot), with
TOPSIS shown as its own row, visually separated at the top (it's a
composite across the other 5, not one of them). One heatmap per protein.

Cell values use the same best-noise-operating-point convention as
make_architecture_comparison_scatter.py: for each (method, protein,
landscape), pick the noise level with the highest mean model_spearman
across seeds, read the metrics there, then average across the 3
landscapes. Skill scores (not raw) are used so every metric sits on the
same chance=0/ideal=1 scale despite having different natural ranges.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import topsis_score, BACKBONE_METRICS, PROTEINS, LANDSCAPES  # noqa: E402

METHOD_DISPLAY = {
    "IG": "Pool First", "PreMLP": "Residual\nPre-MLP",
    "MeanPool": "Per-res. +\nMean Pool", "AttnPool": "Per-res. +\nAttn Pool",
    "OneHot": "One-Hot",
}
METRIC_DISPLAY = {
    "skill_spearman": "Spearman", "skill_auroc": "AUROC",
    "skill_precision_at_k": "Precision@k", "skill_recall_topcat": "Recall\n(top cat.)",
    "skill_sign_acc": "Sign Accuracy",
}
METRICS = list(METRIC_DISPLAY.keys())


def best_noise_values(g, cols):
    by_noise = g.groupby("noise")["model_spearman"].mean()
    if by_noise.empty:
        return None
    best_noise = by_noise.idxmax()
    sub = g[g.noise == best_noise]
    return {c: sub[c].mean() for c in cols}


def main():
    df_all = pd.read_csv("results/synthetic/metric_suite/metric_suite.csv")
    df = df_all[df_all.backbone == "esmc_300m"].copy()
    composite_metrics = [m for m in BACKBONE_METRICS if m in df.columns]
    df["topsis_score"] = topsis_score(df, composite_metrics)
    sweep = df[df.kind == "sweep"]

    methods = list(METHOD_DISPLAY.keys())
    proteins = [p for p in PROTEINS if p in sweep.protein.unique()]
    all_cols = METRICS + ["topsis_score"]

    fig = plt.figure(figsize=(9.5 * len(proteins), 7.0))
    outer = gridspec.GridSpec(1, len(proteins), wspace=0.55)

    for pi, protein in enumerate(proteins):
        p_df = sweep[sweep.protein == protein]
        grid = np.full((len(methods), len(all_cols)), np.nan)

        for mi, method in enumerate(methods):
            m_df = p_df[p_df.method == method]
            if method == "IG":
                m_df = m_df[m_df.seed != 0]
            land_vals = {c: [] for c in all_cols}
            for landscape in LANDSCAPES:
                g = m_df[m_df.landscape == landscape]
                if g.empty:
                    continue
                vals = best_noise_values(g, all_cols)
                if vals is None:
                    continue
                for c in all_cols:
                    land_vals[c].append(vals[c])
            for ci, c in enumerate(all_cols):
                if land_vals[c]:
                    grid[mi, ci] = np.nanmean(land_vals[c])

        # topsis is its own row, separated with a gap above the 5-metric grid
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[pi],
                                                  height_ratios=[1, len(methods)], hspace=0.15)
        ax_top = fig.add_subplot(inner[0])
        ax_main = fig.add_subplot(inner[1])

        topsis_row = grid[:, -1:].T  # (1, n_methods)
        metric_grid = grid[:, :-1].T  # (n_metrics, n_methods)

        vmax = 1.0
        vmin = min(0.0, np.nanmin(metric_grid), np.nanmin(topsis_row))
        cmap = "viridis"

        def _text_color(v):
            frac = (v - vmin) / (vmax - vmin)
            return "white" if frac < 0.6 else "black"

        ax_top.imshow(topsis_row, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for j in range(len(methods)):
            v = topsis_row[0, j]
            if np.isfinite(v):
                ax_top.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=15,
                            color=_text_color(v), fontweight="bold")
        ax_top.set_xticks([])
        ax_top.set_yticks([0])
        ax_top.set_yticklabels(["TOPSIS"], fontsize=15, color="black", fontweight="bold")
        ax_top.set_title(f"{protein}", fontweight="bold", fontsize=19, color="black")

        im = ax_main.imshow(metric_grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(METRICS)):
            for j in range(len(methods)):
                v = metric_grid[i, j]
                if np.isfinite(v):
                    ax_main.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=14,
                                 color=_text_color(v))
        ax_main.set_xticks(range(len(methods)))
        ax_main.set_xticklabels([METHOD_DISPLAY[m] for m in methods], fontsize=13, color="black",
                                 rotation=20, ha="right")
        ax_main.set_yticks(range(len(METRICS)))
        ax_main.set_yticklabels([METRIC_DISPLAY[m] for m in METRICS], fontsize=14, color="black")
        ax_main.tick_params(length=0)
        ax_top.tick_params(length=0)

        cbar = fig.colorbar(im, ax=[ax_top, ax_main], fraction=0.046, pad=0.03)
        cbar.set_label("Skill score (0 = chance, 1 = ideal)", fontsize=15, color="black")
        cbar.ax.tick_params(labelsize=11, labelcolor="black")

    fig.suptitle("Attribution Quality Across Architectures and Metrics (ESM-C 300M)",
                 fontweight="bold", fontsize=19, color="black")

    out = "results/synthetic/metric_suite/architecture_metric_heatmap"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
