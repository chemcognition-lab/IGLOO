#!/usr/bin/env python3
"""
Cascading randomization sanity check, banded across the 3 model-training
seeds (seed_1/2/3, noise=0.0) -- one plot per protein, all 3 landscapes
overlaid, spearman_vs_original vs cascading step. Generates one such figure
per architecture that has full 3-seed cascading-randomization coverage:
IG ("Pool First"), PreMLP, MeanPool. (AttnPool is not included here yet --
its cascading job was submitted separately and isn't done; p53 has no
cascading data yet for any architecture.)

Marks where pooling happens on every panel, with the same shaded-column +
label treatment throughout. Pool First/PreMLP/AttnPool randomize their
pooling layer as its own cascade step (+attention1d / +pool_score). MeanPool
has no such module -- its pooling is a parameter-free mean (see
ESMPerResidueMLP.forward, final_pool='mean') that's applied every step,
right after `final` -- so its band is anchored at the `final` step instead.

Same visual conventions as make_poolfirst_topsis_figure.py (bands, dark
black text/spines, grey provenance note).
"""
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LANDSCAPE_COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}
LANDSCAPE_MARKERS = {"additive": "o", "epistatic": "s", "nonlinear": "D"}
PROTEINS = ["GB1", "T4Lysozyme"]
LANDSCAPES = ["additive", "epistatic", "nonlinear"]
POOLING_MARK_COLOR = "#8a4fff"

# kept in sync with cascading_randomization.py's POOLING_STEP -- duplicated
# here (rather than imported) so this plotting script doesn't need to pull
# in torch/src.igloo just for one small dict.
POOLING_STEP = {
    "Pool First (IG)": "+attention1d",
    "PreMLP": "+attention1d",
    "MeanPool": "final",
    "AttnPool": "+pool_score",
}

# display name -> (source dir, output filename slug)
ARCHITECTURES = {
    "Pool First (IG)": ("results/synthetic/cascading_randomization_baseline", "poolfirst_cascading_by_landscape"),
    "PreMLP": ("results/synthetic/cascading_randomization_premlp", "premlp_cascading_by_landscape"),
    "MeanPool": ("results/synthetic/cascading_randomization_meanpool", "meanpool_cascading_by_landscape"),
    "AttnPool": ("results/synthetic/cascading_randomization_attnpool", "attnpool_cascading_by_landscape"),
}


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


def load_all(src_dir):
    rows = []
    for f in glob.glob(f"{src_dir}/cascading_*__seed_*.csv"):
        m = re.match(r"cascading_(.+)_(additive|epistatic|nonlinear)__seed_(\d+)\.csv",
                     Path(f).name)
        if not m:
            continue
        protein, landscape, seed = m.group(1), m.group(2), int(m.group(3))
        d = pd.read_csv(f)
        d = _drop_noop_steps(d)
        d["protein"] = protein
        d["landscape"] = landscape
        d["seed"] = seed
        rows.append(d)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _mark_pooling_step(ax, levels, display_name):
    """Shade the cascade step that corresponds to the architecture's pooling
    layer -- same band + label treatment for every architecture. MeanPool
    has no randomizable pooling module (parameter-free mean, applied every
    step right after `final`), so its band is anchored at `final` instead."""
    pooling_label = POOLING_STEP.get(display_name)
    if pooling_label is None:
        return
    match = levels[levels.label == pooling_label]
    if match.empty:
        return
    x = match.level.iloc[0]
    ax.axvspan(x - 0.5, x + 0.5, color=POOLING_MARK_COLOR, alpha=0.22, zorder=0)
    ax.text(x, 1.01, "pooling layer", transform=ax.get_xaxis_transform(),
            fontsize=10, color=POOLING_MARK_COLOR, ha="center", va="bottom", fontweight="bold")


def make_figure(display_name, src_dir, out_slug):
    df = load_all(src_dir)
    if df is None:
        print(f"no data found in {src_dir} yet -- skipping {display_name}")
        return

    fig, axes = plt.subplots(1, len(PROTEINS), figsize=(6.5 * len(PROTEINS), 8.0))
    if len(PROTEINS) == 1:
        axes = [axes]

    for ax, protein in zip(axes, PROTEINS):
        p_df = df[df.protein == protein]
        if p_df.empty:
            continue
        levels = p_df[["level", "label"]].drop_duplicates().sort_values("level")
        n_seeds = p_df.groupby("landscape").seed.nunique().max()

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

        _mark_pooling_step(ax, levels, display_name)

        ax.set_xticks(levels.level)
        ax.set_xticklabels(levels.label, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{protein} — {display_name}", fontweight="bold", fontsize=14, color="black", pad=20)
        ax.set_xlabel("Cascading randomization step", fontsize=12, color="black")
        ax.set_ylabel("Spearman ρ vs. original attribution", fontsize=12, color="black")
        ax.tick_params(axis="both", labelsize=10, labelcolor="black", color="black")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        legend = ax.legend(loc="lower left", fontsize=10, title="Landscape", title_fontsize=10)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.text(0.03, 0.33, f"{protein}, ESM-C 300M, {n_seeds} seeds", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.1)

    fig.suptitle(f"Cascading Randomization — {display_name} architecture", fontweight="bold", fontsize=15, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = f"results/synthetic/cascading_randomization_figures/{out_slug}"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png / .pdf")


def main():
    for display_name, (src_dir, out_slug) in ARCHITECTURES.items():
        make_figure(display_name, src_dir, out_slug)


if __name__ == "__main__":
    main()
