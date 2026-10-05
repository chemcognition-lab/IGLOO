#!/usr/bin/env python3
"""
INTERIM version of the Pool First (IG) cascading-randomization figure, using
the existing noise-level data (results/synthetic/cascading_randomization/,
single run per noise level: 0, 0.35, 0.7, 1.0 -- no seeds) since the seeded
version (results/synthetic/cascading_randomization_baseline/, 3 seeds) is
still running. Bands here are mean +/- std ACROSS THE 4 NOISE LEVELS per
landscape, not across seeds -- re-run make_poolfirst_cascading_figure.py
once the seeded job finishes for the real seed-banded version.

Same visual conventions as the other poolfirst_* figures.
"""
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = "results/synthetic/cascading_randomization"
LANDSCAPE_COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}
LANDSCAPE_MARKERS = {"additive": "o", "epistatic": "s", "nonlinear": "D"}
PROTEINS = ["GB1", "T4Lysozyme"]
LANDSCAPES = ["additive", "epistatic", "nonlinear"]


def _drop_noop_steps(d):
    """Some cascade steps are no-ops for architectures that lack that module
    (e.g. 'pre' only exists on PreMLP; 'attention1d'/'pre' don't exist on
    MeanPool) -- randomize_module() silently skips missing attrs, so the
    'randomized' module list is identical to the previous step and the
    measurement is a duplicate (same model state, just re-sampled random
    weights), not a genuine additional degradation step. Drop those rows."""
    d = d.sort_values("level").reset_index(drop=True)
    prev_randomized = d["randomized"].shift(1)
    is_noop = (d["randomized"] == prev_randomized) & d["level"].gt(0)
    return d[~is_noop]


def load_all():
    rows = []
    for f in glob.glob(f"{SRC_DIR}/cascading_*__noise_*.csv"):
        m = re.match(r"cascading_(.+)_(additive|epistatic|nonlinear)__noise_([0-9.]+)\.csv",
                     Path(f).name)
        if not m:
            continue
        protein, landscape, noise = m.group(1), m.group(2), float(m.group(3))
        d = pd.read_csv(f)
        d = _drop_noop_steps(d)
        d["protein"] = protein
        d["landscape"] = landscape
        d["noise"] = noise
        rows.append(d)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def main():
    df = load_all()
    if df is None:
        print(f"no data found in {SRC_DIR}")
        return

    fig, axes = plt.subplots(1, len(PROTEINS), figsize=(6.5 * len(PROTEINS), 8.0))
    if len(PROTEINS) == 1:
        axes = [axes]

    for ax, protein in zip(axes, PROTEINS):
        p_df = df[df.protein == protein]
        if p_df.empty:
            continue
        levels = p_df[["level", "label"]].drop_duplicates().sort_values("level")
        n_noise = p_df.groupby("landscape").noise.nunique().max()

        for landscape in LANDSCAPES:
            g = p_df[p_df.landscape == landscape]
            if g.empty:
                continue
            agg = g.groupby("level").agg(
                mean=("spearman_vs_original", "mean"),
                std=("spearman_vs_original", "std"),
            ).reindex(levels.level).reset_index()

            color = LANDSCAPE_COLORS[landscape]
            marker = LANDSCAPE_MARKERS[landscape]
            ax.plot(agg.level, agg["mean"], f"{marker}-", color=color,
                    markersize=6, linewidth=1.5, alpha=0.95, label=landscape.capitalize())
            ax.fill_between(agg.level, agg["mean"] - agg["std"].fillna(0),
                             agg["mean"] + agg["std"].fillna(0), color=color, alpha=0.2, linewidth=0)

        ax.set_xticks(levels.level)
        ax.set_xticklabels(levels.label, rotation=45, ha="right", fontsize=11)
        ax.set_title(f"{protein} — Pool First (IG)", fontweight="bold", fontsize=17, color="black")
        ax.set_xlabel("Cascading randomization step", fontsize=14, color="black")
        ax.set_ylabel("Spearman ρ vs. original attribution", fontsize=14, color="black")
        ax.tick_params(axis="both", labelsize=12, labelcolor="black", color="black")
        ax.set_ylim(-0.4, 1.05)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        legend = ax.legend(loc="lower left", fontsize=12, title="Landscape", title_fontsize=12)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.text(0.97, 0.97, f"{protein}, ESM-C 300M, band = {n_noise} noise levels (INTERIM, not seeds)",
                transform=ax.transAxes, fontsize=9.5, va="top", ha="right", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

    fig.suptitle("Cascading Randomization — Pool First (IG) architecture [INTERIM: banded across noise, not seeds]",
                 fontweight="bold", fontsize=16, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/synthetic/cascading_randomization_figures/poolfirst_cascading_by_landscape_noisebands_INTERIM"
    fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
