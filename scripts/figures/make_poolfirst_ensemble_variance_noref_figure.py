#!/usr/bin/env python3
"""
Same as make_poolfirst_ensemble_variance_figure.py (std. dev. of TOPSIS
across bootstrap draws vs. ensemble size K, one line per landscape), but
WITHOUT the theoretical 1/sqrt(K) CLT reference curve -- just the observed
spread.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import PROTEINS, LANDSCAPES, BACKBONE_METRICS, topsis_score  # noqa: E402
from make_poolfirst_ensemble_boxplot_figure import (  # noqa: E402
    ensemble_rows, K_VALUES, B_DRAWS, LANDSCAPE_COLORS,
)


def main():
    all_rows = []
    for protein in PROTEINS:
        for landscape in LANDSCAPES:
            all_rows.extend(ensemble_rows(protein, landscape))
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("no seedsweep data found")
        return
    df["topsis"] = topsis_score(df, BACKBONE_METRICS, group_cols=("protein", "landscape"))

    proteins = [p for p in PROTEINS if p in df.protein.unique()]
    fig, axes = plt.subplots(1, len(proteins), figsize=(7.0 * len(proteins), 7.0))
    if len(proteins) == 1:
        axes = [axes]

    for ax, protein in zip(axes, proteins):
        p_df = df[df.protein == protein]

        for landscape in LANDSCAPES:
            g = p_df[p_df.landscape == landscape]
            if g.empty:
                continue
            color = LANDSCAPE_COLORS[landscape]
            std_by_k = g.groupby("K")["topsis"].std().reindex(K_VALUES)
            ax.plot(K_VALUES, std_by_k.to_numpy(), "o-", color=color, markersize=7,
                    linewidth=2.0, alpha=0.95, label=landscape.capitalize(), zorder=3)

        ax.set_xticks(K_VALUES)
        ax.set_title(f"{protein} — Pool First (IG)", fontweight="bold", fontsize=17, color="black")
        ax.set_xlabel("Number of seeds averaged into ensemble (K)", fontsize=13.5, color="black")
        ax.set_ylabel("SD of TOPSIS", fontsize=13, color="black")
        ax.tick_params(axis="both", labelsize=12, labelcolor="black", color="black")
        ax.set_ylim(bottom=0)

        legend = ax.legend(loc="upper right", fontsize=12, title="Landscape", title_fontsize=12)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")

        ax.text(0.03, 0.03, f"{protein}, ESM-C 300M, {B_DRAWS} bootstrap draws/K (with replacement)",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.2)

    fig.suptitle("Ensemble Variance Reduction — Pool First (IG) architecture",
                 fontweight="bold", fontsize=18, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/synthetic/metric_suite/poolfirst_ensemble_variance_by_landscape_noref"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
