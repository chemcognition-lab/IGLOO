#!/usr/bin/env python3
"""
TOPSIS vs. backbone size: x-axis = each backbone as its own categorical
position (grouped by family, with a gap between families), labeled with its
parameter count; y-axis = TOPSIS score. Individual data points (pooled
across protein x architecture x landscape) are shown as a jittered spread
per backbone, with a bold mean marker, colored by model family. Means
within a family are connected by a line (in size order) so the
within-family scaling trend is visible; families are never connected to
each other.

Backbones get one slot each on the x-axis (no shared numeric scale), so
models that happen to share a parameter count (ESM-C 300M/600M vs. E1
150M/300M/600M) never overlap -- each is simply its own tick.

Pools across all proteins that have data for a given backbone (GB1 and
T4Lysozyme have all 7 backbones; p53 only has ankh_base, ankh_large,
esmc_600m) -- so point counts differ by backbone (18 vs 27), but every
backbone's spread is the full available n, not a subsample.

Parameter counts: ESM-C 300M/600M and Profluent E1 150M/300M/600M are given
by their model names directly. Ankh base/large are 453.3M / 1.15B
parameters per the Ankh paper (ElNaggar et al. 2023) and HuggingFace model
cards -- not inferable from the name, so recorded explicitly below.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

FAMILY_ORDER = ["ESM-C", "Ankh", "Profluent E1"]
FAMILY = {
    "esmc_300m": "ESM-C", "esmc_600m": "ESM-C",
    "ankh_base": "Ankh", "ankh_large": "Ankh",
    "e1_150m": "Profluent E1", "e1_300m": "Profluent E1", "e1_600m": "Profluent E1",
}
SIZE_M = {  # millions of parameters
    "esmc_300m": 300, "esmc_600m": 600,
    "ankh_base": 453.3, "ankh_large": 1150,
    "e1_150m": 150, "e1_300m": 300, "e1_600m": 600,
}
SIZE_LABEL = {
    "esmc_300m": "300M", "esmc_600m": "600M",
    "ankh_base": "453M", "ankh_large": "1.15B",
    "e1_150m": "150M", "e1_300m": "300M", "e1_600m": "600M",
}
FAMILY_COLORS = {"ESM-C": "#0072B2", "Ankh": "#E69F00", "Profluent E1": "#009E73"}
FAMILY_MARKERS = {"ESM-C": "o", "Ankh": "s", "Profluent E1": "D"}
GROUP_GAP = 1.0  # extra x-spacing inserted between families


def main():
    bt = pd.read_csv("results/synthetic/metric_suite/backbone_topsis_cells.csv")
    topsis = bt[bt.metric == "topsis"]

    rng = np.random.default_rng(0)

    # assign one categorical x slot per backbone, grouped by family with a
    # gap between groups, backbones within a family ordered by size
    x_pos = {}
    cursor = 0.0
    for fam in FAMILY_ORDER:
        fam_backbones = sorted([b for b in FAMILY if FAMILY[b] == fam], key=lambda b: SIZE_M[b])
        for b in fam_backbones:
            x_pos[b] = cursor
            cursor += 1.0
        cursor += GROUP_GAP

    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    family_points = {fam: {"x": [], "y": []} for fam in FAMILY_COLORS}

    for b, x0 in x_pos.items():
        vals = topsis[topsis.backbone == b]["value"].to_numpy()
        if len(vals) == 0:
            continue
        fam = FAMILY[b]
        color = FAMILY_COLORS[fam]
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(x0 + jitter, vals, color=color, alpha=0.35, s=30,
                   edgecolors="none", zorder=2)
        mean_v = vals.mean()
        ax.scatter(x0, mean_v, color=color, s=170, marker=FAMILY_MARKERS[fam],
                   edgecolors="black", linewidths=1.2, zorder=4)
        ax.vlines(x0, vals.min(), vals.max(), color=color, alpha=0.3, linewidth=1.3, zorder=1)
        family_points[fam]["x"].append(x0)
        family_points[fam]["y"].append(mean_v)

    for fam, pts in family_points.items():
        if len(pts["x"]) < 2:
            continue
        order = np.argsort(pts["x"])
        xs = np.array(pts["x"])[order]
        ys = np.array(pts["y"])[order]
        ax.plot(xs, ys, "-", color=FAMILY_COLORS[fam], linewidth=2.0, alpha=0.8, zorder=3)

    backbones_present = [b for b in x_pos if not topsis[topsis.backbone == b].empty]
    ax.set_xticks([x_pos[b] for b in backbones_present])
    ax.set_xticklabels([SIZE_LABEL[b] for b in backbones_present], fontsize=9, color="black")
    ax.set_xlim(-0.6, cursor - GROUP_GAP + 0.6)

    # family group labels under the size ticks
    for fam in FAMILY_ORDER:
        fam_x = [x_pos[b] for b in x_pos if FAMILY[b] == fam and b in backbones_present]
        if not fam_x:
            continue
        ax.text(np.mean(fam_x), -0.085, fam, transform=ax.get_xaxis_transform(),
                fontsize=9.5, fontweight="bold", color=FAMILY_COLORS[fam], ha="center", va="top")

    ax.set_xlabel("Backbone (grouped by model family, labeled by parameter count)",
                  fontsize=10, color="black", labelpad=30)
    ax.set_ylabel("TOPSIS Score", fontsize=10, color="black")
    ax.set_title("TOPSIS Score vs. pLM Backbone Size", fontweight="bold", fontsize=13, color="black")
    ax.tick_params(axis="both", labelsize=9, labelcolor="black", color="black")
    ax.set_ylim(-0.02, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.0)

    handles = [plt.Line2D([0], [0], marker=FAMILY_MARKERS[fam], color=FAMILY_COLORS[fam],
                           markerfacecolor=FAMILY_COLORS[fam], markeredgecolor="black",
                           markersize=9, linewidth=1.5, label=fam)
               for fam in FAMILY_COLORS]
    legend = ax.legend(handles=handles, title="Model family", loc="lower right", fontsize=9, title_fontsize=9)
    legend.get_frame().set_edgecolor("black")
    for text in legend.get_texts():
        text.set_color("black")
    legend.get_title().set_color("black")

    ax.text(0.02, 0.03, "Points pooled across protein × architecture × landscape\n"
                         "(GB1, T4Lysozyme: all 7 backbones; p53: ankh_base/large, esmc_600m only)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color="#666666", style="italic")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    out = "results/synthetic/metric_suite/backbone_size_scaling"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
