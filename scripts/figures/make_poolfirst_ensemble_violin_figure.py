#!/usr/bin/env python3
"""
Violin-plot version of make_poolfirst_ensemble_boxplot_figure.py: same
bootstrap-with-replacement TOPSIS draws (K=1..5 seeds averaged into an
ensemble), but showing the full estimated density per K instead of just
box/whisker quartiles -- makes it easier to see when a distribution is
multimodal (e.g. a persistent low-outlier cluster from a single
divergent seed being repeatedly redrawn) rather than just "has long
whiskers."
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
    fig, axes = plt.subplots(1, len(proteins), figsize=(7.5 * len(proteins), 8.0))
    if len(proteins) == 1:
        axes = [axes]

    n_landscapes = len(LANDSCAPES)
    offsets = np.linspace(-0.25, 0.25, n_landscapes)
    violin_width = 0.9 / n_landscapes * 0.85

    for ax, protein in zip(axes, proteins):
        p_df = df[df.protein == protein]

        for li, landscape in enumerate(LANDSCAPES):
            g = p_df[p_df.landscape == landscape]
            if g.empty:
                continue
            color = LANDSCAPE_COLORS[landscape]
            data, positions = [], []
            for K in K_VALUES:
                vals = g[g.K == K]["topsis"].to_numpy()
                if len(vals) < 2 or np.allclose(vals, vals[0]):
                    continue
                data.append(vals)
                positions.append(K + offsets[li])
            if not data:
                continue

            vp = ax.violinplot(data, positions=positions, widths=violin_width,
                               showmedians=True, showextrema=True)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.5)
                body.set_linewidth(0.8)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                vp[key].set_color("black")
                vp[key].set_linewidth(1.0 if key != "cmedians" else 1.6)

        handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=LANDSCAPE_COLORS[l],
                              markeredgecolor="black", markersize=11, label=l.capitalize())
                  for l in LANDSCAPES]
        legend = ax.legend(handles=handles, loc="lower right", fontsize=12, title="Landscape", title_fontsize=12)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")

        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=12, color="black")
        ax.set_xlim(0.4, max(K_VALUES) + 0.6)
        ax.set_title(f"{protein} — Pool First (IG)", fontweight="bold", fontsize=17, color="black")
        ax.set_xlabel("Number of seeds averaged into ensemble (K)", fontsize=13.5, color="black")
        ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=14, color="black")
        ax.tick_params(axis="both", labelsize=12, labelcolor="black", color="black")
        ax.set_ylim(-0.02, 1.05)
        ax.text(0.03, 0.97, f"{protein}, ESM-C 300M, {B_DRAWS} bootstrap draws/K (with replacement)",
                transform=ax.transAxes, fontsize=10, va="top", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.2)

    fig.suptitle("TOPSIS Score vs. Ensemble Size — Pool First (IG) architecture",
                 fontweight="bold", fontsize=18, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/synthetic/metric_suite/poolfirst_ensemble_violin_by_landscape"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
