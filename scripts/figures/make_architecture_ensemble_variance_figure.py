#!/usr/bin/env python3
"""
Ensemble-size variance reduction (std. dev. of bootstrap TOPSIS vs K, same
bootstrap-with-replacement methodology as make_poolfirst_ensemble_variance_figure.py),
generalized across PreMLP, MeanPool, and AttnPool -- filling the gap where
only Pool First (IG, K=1..5) had this analysis; these three only have 3
seeds each (K=1..3), same checkpoints already used for cascading
randomization (noise_0.0/seed_{1,2,3}).

One standalone figure per architecture (matches Pool First's own standalone
poolfirst_ensemble_variance_by_landscape.png), one panel per protein with
data (GB1, T4Lysozyme; p53 pending until its esmc_300m seedsweep exists).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import PROTEINS, LANDSCAPES, BACKBONE_METRICS, topsis_score, all_metrics, _model_perf  # noqa: E402

LANDSCAPE_COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}
SEEDS = [1, 2, 3]
K_VALUES = [1, 2, 3]
B_DRAWS = 300

# display name -> (checkpoint dir, output filename slug)
ARCHS = {
    "PreMLP": ("results/synthetic/premlp", "premlp_ensemble_variance_by_landscape"),
    "MeanPool": ("results/synthetic/newarch_perres_mean", "meanpool_ensemble_variance_by_landscape"),
    "AttnPool": ("results/synthetic/newarch_perres_attn", "attnpool_ensemble_variance_by_landscape"),
}


def _load_seed_vectors(src_dir, protein, landscape):
    combo_dir = Path(src_dir) / f"{protein}_{landscape}"
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


def ensemble_rows(src_dir, protein, landscape):
    by_noise = _load_seed_vectors(src_dir, protein, landscape)
    if not by_noise:
        return []
    noise_means = {n: np.nanmean([v[2] for v in seeds.values()]) for n, seeds in by_noise.items()}
    noise_means = {n: v for n, v in noise_means.items() if np.isfinite(v)}
    if not noise_means:
        return []
    best_noise = max(noise_means, key=noise_means.get)
    seeds = by_noise[best_noise]
    available = sorted(s for s in SEEDS if s in seeds)
    if len(available) < max(K_VALUES):
        return []

    rng = np.random.default_rng(hash((src_dir, protein, landscape)) & 0xFFFFFFFF)
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


def make_figure(display_name, src_dir, out_slug):
    all_rows = []
    for protein in PROTEINS:
        for landscape in LANDSCAPES:
            all_rows.extend(ensemble_rows(src_dir, protein, landscape))
    df = pd.DataFrame(all_rows)
    if df.empty:
        print(f"no seedsweep data found for {display_name} -- skipping")
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
            std1 = std_by_k.loc[1]
            if np.isfinite(std1):
                ref = std1 / np.sqrt(np.array(K_VALUES, dtype=float))
                ax.plot(K_VALUES, ref, "--", color=color, linewidth=1.3, alpha=0.55, zorder=1)

        ax.set_xticks(K_VALUES)
        ax.set_title(f"{protein} — {display_name}", fontweight="bold", fontsize=17, color="black")
        ax.set_xlabel("Number of seeds averaged into ensemble (K)", fontsize=13.5, color="black")
        ax.set_ylabel("SD of TOPSIS", fontsize=13, color="black")
        ax.tick_params(axis="both", labelsize=12, labelcolor="black", color="black")
        ax.set_ylim(bottom=0)
        legend = ax.legend(loc="upper right", fontsize=12, title="Landscape", title_fontsize=12)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.add_artist(legend)
        style_handles = [
            plt.Line2D([0], [0], color="#444444", marker="o", linestyle="-", lw=2.0, ms=7, label="Observed std. dev."),
            plt.Line2D([0], [0], color="#444444", linestyle="--", lw=1.3, label="Theoretical 1/√K (CLT)"),
        ]
        style_legend = ax.legend(handles=style_handles, loc="upper left", fontsize=10, framealpha=0.9)
        style_legend.get_frame().set_edgecolor("black")
        for text in style_legend.get_texts():
            text.set_color("black")
        ax.text(0.03, 0.03, f"{protein}, ESM-C 300M, {B_DRAWS} bootstrap draws/K (with replacement)",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.2)

    fig.suptitle(f"Ensemble Variance Reduction — {display_name} architecture",
                 fontweight="bold", fontsize=18, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = f"results/synthetic/metric_suite/{out_slug}"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png / .pdf")


def main():
    for display_name, (src_dir, out_slug) in ARCHS.items():
        make_figure(display_name, src_dir, out_slug)


if __name__ == "__main__":
    main()
