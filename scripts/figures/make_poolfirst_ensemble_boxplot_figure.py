#!/usr/bin/env python3
"""
TOPSIS Attribution Quality vs. ensemble size (K): for the Pool First (IG)
architecture at esmc_300m, shows how averaging more independently-seeded
models' raw IG attribution maps into one ensemble reduces the variance of
the resulting attribution quality. Same visual style as
make_poolfirst_topsis_figure.py (one narrow subplot per protein, landscape
colors, dark black spines/text, grey provenance note), but rendered as
boxplots (K on x-axis) instead of a line/band.

Methodology, per (protein, landscape):
  1. Pick the single best-noise operating point (the noise level that
     maximizes mean model_spearman across the 5 seedsweep seeds) -- same
     convention used throughout this project.
  2. At that noise, load each of the 5 seeds' raw per-position attribution
     vectors (magnitude + signed gradient).
  3. For K = 1..5, draw B_DRAWS bootstrap samples of size K *with
     replacement* from the 5 seeds, average each draw's attribution vectors
     elementwise into one ensemble attribution, and recompute the 5
     shortlisted metrics on that ensemble.
  4. TOPSIS-rank all of these ensemble alternatives together within the
     (protein, landscape) group (same topsis_score() procedure as
     everywhere else in this project).

Deliberately NOT exhaustive C(5,K) enumeration without replacement: that
design only has 5/10/10/5/1 distinct, increasingly-overlapping subsets as
K -> 5, and collapses to exactly one subset (hence exactly zero variance)
at K=5 purely as a combinatorial artifact of having only 5 seeds total --
not because the ensemble is actually perfectly denoised. Bootstrap
resampling with replacement gives B_DRAWS independent (if correlated)
draws at every K including K=5, so the variance only approaches zero
asymptotically, which is the honest way to show the law-of-large-numbers
denoising trend without conflating it with "we ran out of distinct
subsets."
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import (  # noqa: E402
    PROTEINS, LANDSCAPES, BACKBONE_METRICS, topsis_score, all_metrics, _model_perf,
)

SRC = Path("results/synthetic/seedsweep")
SEEDS = [1, 2, 3, 4, 5]
K_VALUES = [1, 2, 3, 4, 5]
B_DRAWS = 300  # bootstrap draws per (protein, landscape, K)
LANDSCAPE_COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}


def _load_seed_vectors(protein, landscape):
    """noise -> seed -> (mag, sgn, model_spearman)."""
    combo_dir = SRC / f"{protein}_{landscape}"
    out = {}
    if not combo_dir.is_dir():
        return out
    for nd in sorted(combo_dir.glob("noise_*")):
        noise = float(nd.name.split("_", 1)[1])
        seeds = {}
        for sd in sorted(nd.glob("seed_*")):
            seed = int(sd.name.split("_", 1)[1])
            leaf = sd / "linear"
            f = list(leaf.glob("*_position_importance_summary.csv"))
            if not f:
                continue
            d = pd.read_csv(f[0]).sort_values("position")
            mag = d["importance_score"].to_numpy()
            sgn = d["mean_gradient"].to_numpy() if "mean_gradient" in d else mag
            mp = _model_perf(leaf)
            seeds[seed] = (mag, sgn, mp)
        if seeds:
            out[noise] = seeds
    return out


def ensemble_rows(protein, landscape):
    by_noise = _load_seed_vectors(protein, landscape)
    if not by_noise:
        return []
    # best noise = highest mean model_spearman across the 5 seeds
    noise_means = {n: np.nanmean([v[2] for v in seeds.values()]) for n, seeds in by_noise.items()}
    noise_means = {n: v for n, v in noise_means.items() if np.isfinite(v)}
    if not noise_means:
        return []
    best_noise = max(noise_means, key=noise_means.get)
    seeds = by_noise[best_noise]
    available = sorted(s for s in SEEDS if s in seeds)
    if len(available) < max(K_VALUES):
        return []

    rng = np.random.default_rng(hash((protein, landscape)) & 0xFFFFFFFF)
    rows = []
    for K in K_VALUES:
        for _ in range(B_DRAWS):
            combo = rng.choice(available, size=K, replace=True)
            mags = np.mean([seeds[s][0] for s in combo], axis=0)
            sgns = np.mean([seeds[s][1] for s in combo], axis=0)
            met = all_metrics(mags, protein, signed=sgns)
            row = {"protein": protein, "landscape": landscape, "K": K}
            row.update({m: met.get(m, np.nan) for m in BACKBONE_METRICS})
            rows.append(row)
    return rows


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
    box_width = 0.9 / n_landscapes * 0.8

    for ax, protein in zip(axes, proteins):
        p_df = df[df.protein == protein]
        rng = np.random.default_rng(0)

        for li, landscape in enumerate(LANDSCAPES):
            g = p_df[p_df.landscape == landscape]
            if g.empty:
                continue
            color = LANDSCAPE_COLORS[landscape]
            data, positions = [], []
            for K in K_VALUES:
                vals = g[g.K == K]["topsis"].to_numpy()
                if len(vals) == 0:
                    continue
                data.append(vals)
                positions.append(K + offsets[li])

            bp = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=True,
                            showfliers=False, zorder=3,
                            medianprops=dict(color="black", linewidth=1.6),
                            boxprops=dict(facecolor=color, alpha=0.45, edgecolor="black", linewidth=1.0),
                            whiskerprops=dict(color="black", linewidth=1.0),
                            capprops=dict(color="black", linewidth=1.0))
            for pos, vals in zip(positions, data):
                show = vals if len(vals) <= 60 else rng.choice(vals, size=60, replace=False)
                jitter = rng.uniform(-box_width * 0.3, box_width * 0.3, size=len(show))
                ax.scatter(np.full(len(show), pos) + jitter, show, color=color, alpha=0.35,
                          s=12, edgecolors="none", zorder=4)

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
                transform=ax.transAxes,
                fontsize=10, va="top", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.2)

    fig.suptitle("TOPSIS Score vs. Ensemble Size — Pool First (IG) architecture",
                 fontweight="bold", fontsize=18, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/synthetic/metric_suite/poolfirst_ensemble_boxplot_by_landscape"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
