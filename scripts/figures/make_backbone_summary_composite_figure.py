#!/usr/bin/env python3
"""
Composite pLM-backbone comparison figure, sized to fit a 183 mm x 170 mm
journal page (both are hard maxima -- figsize is in inches, 183/25.4 x
170/25.4). Two rows, one column per protein (GB1, T4Lysozyme, p53):

  Row A (top):    backbone_topsis_strip.py's panel -- TOPSIS spread per
                  backbone (individual arch x landscape points + mean),
                  colored by model family.
  Row B (bottom): the arch x backbone TOPSIS heatmap
                  (metric_suite.py's plot_arch_backbone_heatmap /
                  metric_topsis_heatmap.png), but aggregated ACROSS
                  LANDSCAPES per cell (mean of the 3 landscape values) with
                  the landscape-to-landscape spread shown as a second
                  "+-std" line of cell text (a heatmap has no literal error
                  bars, so this is the standard way to carry the same
                  uncertainty information), using viridis (the original
                  used inferno) and the project's standard architecture
                  display names (make_architecture_metric_heatmap.py's
                  METHOD_DISPLAY) instead of the raw PreMLP/MeanPool/AttnPool
                  keys.

Same backbone row order top-to-bottom in both rows of a given protein's
column, so the two rows read as one connected picture: the top row's
per-backbone spread, broken down by which architecture contributes what in
the row below.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import PROTEINS, BACKBONE_ARCHS  # noqa: E402
from make_backbone_topsis_strip import (  # noqa: E402
    FAMILY, FAMILY_COLORS, BACKBONE_ORDER, BACKBONE_DISPLAY,
)

MM_PER_IN = 25.4
MAX_WIDTH_IN = 183 / MM_PER_IN
MAX_HEIGHT_IN = 170 / MM_PER_IN

ARCH_DISPLAY = {
    "PreMLP": "Residual\nPre-MLP", "MeanPool": "Per-res. +\nMean Pool",
    "AttnPool": "Per-res. +\nAttn Pool",
}

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, CELL_FS = 8, 6.5, 5.5, 5.5, 5.5


def _strip_panel(ax, p_df, backbones, rng, show_ylabel_ticks, show_xlabel):
    y_pos = {b: i for i, b in enumerate(backbones)}
    for b in backbones:
        vals = p_df[p_df.backbone == b]["value"].to_numpy()
        y = y_pos[b]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        color = FAMILY_COLORS[FAMILY[b]]
        ax.scatter(vals, np.full(len(vals), y) + jitter, color=color, alpha=0.45,
                   s=10, edgecolors="none", zorder=2)
        ax.scatter(vals.mean(), y, color=color, s=38, marker="D",
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.hlines(y, vals.min(), vals.max(), color=color, alpha=0.25, linewidth=0.9, zorder=1)

    ax.set_yticks(range(len(backbones)))
    if show_ylabel_ticks:
        ax.set_yticklabels([BACKBONE_DISPLAY[b] for b in backbones], fontsize=TICK_FS, color="black")
    else:
        ax.set_yticklabels([])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(backbones) - 0.4)
    if show_xlabel:
        ax.set_xlabel("TOPSIS Score", fontsize=LABEL_FS, color="black")
    ax.tick_params(axis="both", labelsize=TICK_FS, labelcolor="black", color="black", width=0.5, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.invert_yaxis()


def _heatmap_panel(ax, p_df, backbones, archs, show_ylabel_ticks):
    grid_mean = np.full((len(backbones), len(archs)), np.nan)
    grid_std = np.full((len(backbones), len(archs)), np.nan)
    for i, b in enumerate(backbones):
        for j, a in enumerate(archs):
            sub = p_df[(p_df.backbone == b) & (p_df.arch == a)]["value"]
            if sub.empty:
                continue
            grid_mean[i, j] = sub.mean()
            grid_std[i, j] = sub.std()

    im = ax.imshow(grid_mean, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    for i in range(len(backbones)):
        for j in range(len(archs)):
            v, s = grid_mean[i, j], grid_std[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v < 0.6 else "black"
            label = f"{v:.2f}" + (f"\n±{s:.2f}" if np.isfinite(s) else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=CELL_FS, color=txt_color, linespacing=1.3)

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([ARCH_DISPLAY[a] for a in archs], fontsize=TICK_FS, color="black")
    ax.set_yticks(range(len(backbones)))
    if show_ylabel_ticks:
        ax.set_yticklabels([BACKBONE_DISPLAY[b] for b in backbones], fontsize=TICK_FS, color="black")
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    return im


def main():
    bt = pd.read_csv("results/synthetic/metric_suite/backbone_topsis_cells.csv")
    topsis = bt[bt.metric == "topsis"]
    archs = [a for a in BACKBONE_ARCHS if a in topsis.arch.unique()]

    proteins = [p for p in PROTEINS if p in topsis.protein.unique()]
    fig, axes = plt.subplots(2, len(proteins), figsize=(MAX_WIDTH_IN, MAX_HEIGHT_IN * 0.97),
                             gridspec_kw={"height_ratios": [0.85, 1.0]}, layout="constrained")

    rng = np.random.default_rng(0)
    im = None
    reference_backbones = None
    for col, protein in enumerate(proteins):
        p_df = topsis[topsis.protein == protein]
        backbones = [b for b in BACKBONE_ORDER if b in p_df.backbone.unique()]
        ax_top, ax_bot = axes[0, col], axes[1, col]

        # Only skip this column's own y-tick labels when its backbone list is
        # IDENTICAL to the reference (leftmost) column's -- e.g. GB1 and
        # T4Lysozyme both have all 7 backbones, so repeating the same labels
        # a 2nd/3rd time is pure redundancy. p53 has only 3 of the 7, so its
        # rows get rescaled to fill the same panel height and would silently
        # misalign with GB1's label positions if left unlabeled -- a real
        # mislabeling risk, not just visual clutter, so it always gets its
        # own labels when its backbone set differs.
        if reference_backbones is None:
            reference_backbones = backbones
        show_labels = (col == 0) or (backbones != reference_backbones)

        _strip_panel(ax_top, p_df, backbones, rng, show_ylabel_ticks=show_labels,
                     show_xlabel=(col == len(proteins) // 2))
        ax_top.set_title(protein, fontweight="bold", fontsize=TITLE_FS, color="black")

        im = _heatmap_panel(ax_bot, p_df, backbones, archs, show_ylabel_ticks=show_labels)

    axes[0, 0].text(-0.14, 1.18, "A", transform=axes[0, 0].transAxes, fontsize=11,
                    fontweight="bold", color="black", va="top", ha="left")
    axes[1, 0].text(-0.14, 1.08, "B", transform=axes[1, 0].transAxes, fontsize=11,
                    fontweight="bold", color="black", va="top", ha="left")

    # constrained layout doesn't know fig.suptitle() and a plain fig.legend()
    # need separate vertical slots, so carve out a blank top margin by hand
    # (10% of figure height) and place both inside it, suptitle above the
    # legend, rather than letting them fight over the same "outside" slot.
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))

    fig.suptitle("pLM Backbone Comparison — Spread (top) and Architecture Breakdown (bottom)",
                 fontweight="bold", fontsize=TITLE_FS + 1, color="black", y=0.995)

    handles = [plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=c,
                           markeredgecolor="black", markersize=6, label=fam)
               for fam, c in FAMILY_COLORS.items()]
    fig.legend(handles=handles, title="Model family", loc="upper right",
               bbox_to_anchor=(0.995, 0.78), ncol=1, fontsize=LEGEND_FS, title_fontsize=LEGEND_FS,
               frameon=True, edgecolor="black")

    cbar = fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.85, pad=0.04, fraction=0.025)
    cbar.set_label("TOPSIS Score (mean across landscapes)", fontsize=LABEL_FS, color="black")
    cbar.ax.tick_params(labelsize=TICK_FS)

    out = "results/synthetic/metric_suite/backbone_summary_composite"
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf  ({MAX_WIDTH_IN:.3f} in x {MAX_HEIGHT_IN*0.97:.3f} in = "
          f"{MAX_WIDTH_IN*MM_PER_IN:.0f} mm x {MAX_HEIGHT_IN*0.97*MM_PER_IN:.0f} mm)")


if __name__ == "__main__":
    main()
