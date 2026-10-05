#!/usr/bin/env python3
"""
TOPSIS spread per pLM backbone: y-axis = backbone, x-axis = TOPSIS score,
individual data points (3 architectures x 3 landscapes = 9 per backbone)
plus the mean, colored by model family. Three narrow subplots side by side,
one per protein (GB1, T4Lysozyme, p53) -- p53 only has 3 of the 7 backbones
(ankh_base, ankh_large, esmc_600m), since it was never run on esmc_300m or
the Profluent E1 family.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import PROTEINS  # noqa: E402

FAMILY = {
    "esmc_300m": "ESM-C", "esmc_600m": "ESM-C",
    "ankh_base": "Ankh", "ankh_large": "Ankh",
    "e1_150m": "Profluent E1", "e1_300m": "Profluent E1", "e1_600m": "Profluent E1",
}
FAMILY_COLORS = {"ESM-C": "#0072B2", "Ankh": "#E69F00", "Profluent E1": "#009E73"}
BACKBONE_ORDER = ["esmc_300m", "esmc_600m", "ankh_base", "ankh_large", "e1_150m", "e1_300m", "e1_600m"]
BACKBONE_DISPLAY = {
    "esmc_300m": "ESM-C 300M", "esmc_600m": "ESM-C 600M",
    "ankh_base": "Ankh-base", "ankh_large": "Ankh-large",
    "e1_150m": "E1-150M", "e1_300m": "E1-300M", "e1_600m": "E1-600M",
}


def main():
    bt = pd.read_csv("results/synthetic/metric_suite/backbone_topsis_cells.csv")
    topsis = bt[bt.metric == "topsis"]

    proteins = [p for p in PROTEINS if p in topsis.protein.unique()]
    fig, axes = plt.subplots(1, len(proteins), figsize=(4.2 * len(proteins), 7.0), sharex=True)
    if len(proteins) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)

    for ax, protein in zip(axes, proteins):
        p_df = topsis[topsis.protein == protein]
        backbones = [b for b in BACKBONE_ORDER if b in p_df.backbone.unique()]
        y_pos = {b: i for i, b in enumerate(backbones)}

        for b in backbones:
            vals = p_df[p_df.backbone == b]["value"].to_numpy()
            y = y_pos[b]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            color = FAMILY_COLORS[FAMILY[b]]
            ax.scatter(vals, np.full(len(vals), y) + jitter, color=color, alpha=0.45,
                       s=35, edgecolors="none", zorder=2)
            ax.scatter(vals.mean(), y, color=color, s=130, marker="D",
                       edgecolors="black", linewidths=1.0, zorder=3)
            ax.hlines(y, vals.min(), vals.max(), color=color, alpha=0.25, linewidth=1.2, zorder=1)

        ax.set_yticks(range(len(backbones)))
        ax.set_yticklabels([BACKBONE_DISPLAY[b] for b in backbones], fontsize=9.5, color="black")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.6, len(backbones) - 0.4)
        ax.set_title(protein, fontweight="bold", fontsize=14, color="black")
        ax.set_xlabel("TOPSIS Score", fontsize=11, color="black")
        ax.tick_params(axis="both", labelsize=9.5, labelcolor="black", color="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.0)
        ax.invert_yaxis()

    handles = [plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=c,
                           markeredgecolor="black", markersize=10, label=fam)
               for fam, c in FAMILY_COLORS.items()]
    fig.legend(handles=handles, title="Model family", loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9.5, title_fontsize=9.5,
               frameon=True, edgecolor="black")

    fig.suptitle("TOPSIS Score by pLM Backbone (individual points + mean, across 3 architectures × 3 landscapes)",
                 fontweight="bold", fontsize=15, color="black")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    out = "results/synthetic/metric_suite/backbone_topsis_strip"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
