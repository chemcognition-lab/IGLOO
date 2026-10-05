#!/usr/bin/env python3
"""
Publication/print-ready version of make_cascading_by_landscape_figure.py:
sized to a fixed 183 mm total width (a common journal full-width figure
spec) and shows one panel per protein, side by side -- 2 panels now
(GB1, T4Lysozyme), automatically becoming 3 once p53 cascading-randomization
data exists (PRINT_PROTEIN_ORDER already includes it; it's just filtered out
until the data is there).

183 mm = 7.205 in (matplotlib figsize is always in inches). Reusing the
on-screen version's font sizes at that width would be illegible -- each
panel only gets ~1/Nth of 7.2 in instead of the screen version's fixed 6.5in
per panel, roughly a 2.7x reduction at N=3 -- so every font size, marker,
and line width here is scaled down to a journal-appropriate point size
(kept just above the ~6pt floor most journals enforce at final print size),
not just a smaller canvas.

Reuses load_all()/_mark_pooling_step()/POOLING_STEP/ARCHITECTURES etc. from
make_cascading_by_landscape_figure.py so both versions stay in sync.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from make_cascading_by_landscape_figure import (  # noqa: E402
    ARCHITECTURES, LANDSCAPE_COLORS, LANDSCAPE_MARKERS, LANDSCAPES,
    POOLING_MARK_COLOR, POOLING_STEP, load_all,
)

MM_PER_IN = 25.4
TOTAL_WIDTH_MM = 183
TOTAL_WIDTH_IN = TOTAL_WIDTH_MM / MM_PER_IN  # 7.205 in
PANEL_HEIGHT_IN = 2.7

PRINT_PROTEIN_ORDER = ["GB1", "T4Lysozyme", "p53"]


def _mark_pooling_step_print(ax, levels, display_name):
    pooling_label = POOLING_STEP.get(display_name)
    if pooling_label is None:
        return
    match = levels[levels.label == pooling_label]
    if match.empty:
        return
    x = match.level.iloc[0]
    ax.axvspan(x - 0.5, x + 0.5, color=POOLING_MARK_COLOR, alpha=0.22, zorder=0)
    ax.text(x, 1.02, "pooling", transform=ax.get_xaxis_transform(),
            fontsize=6, color=POOLING_MARK_COLOR, ha="center", va="bottom", fontweight="bold")


def make_figure(display_name, src_dir, out_slug):
    df = load_all(src_dir)
    if df is None:
        print(f"no data found in {src_dir} yet -- skipping {display_name}")
        return

    proteins = [p for p in PRINT_PROTEIN_ORDER if p in df.protein.unique()]
    n = len(proteins)
    fig, axes = plt.subplots(1, n, figsize=(TOTAL_WIDTH_IN, PANEL_HEIGHT_IN))
    if n == 1:
        axes = [axes]

    for ax, protein in zip(axes, proteins):
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
                    markersize=2.8, linewidth=0.9, alpha=0.95, label=landscape.capitalize())
            ax.fill_between(agg.level, agg["mean"] - agg["std"].fillna(0),
                             agg["mean"] + agg["std"].fillna(0), color=color, alpha=0.2, linewidth=0)

        _mark_pooling_step_print(ax, levels, display_name)

        ax.set_xticks(levels.level)
        ax.set_xticklabels(levels.label, rotation=45, ha="right", fontsize=5)
        ax.set_title(f"{protein} — {display_name}", fontweight="bold", fontsize=7, color="black", pad=10)
        ax.set_xlabel("Cascading step", fontsize=6, color="black")
        ax.set_ylabel("Spearman ρ vs. original", fontsize=6, color="black")
        ax.tick_params(axis="both", labelsize=5, labelcolor="black", color="black", width=0.5, length=2)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="#999999", linewidth=0.5, linestyle="--")
        legend = ax.legend(loc="lower left", fontsize=5, title="Landscape", title_fontsize=5,
                           handlelength=1.3, labelspacing=0.3, borderpad=0.4)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.5)
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.text(0.04, 0.37, f"{protein}, ESM-C 300M, {n_seeds} seeds", transform=ax.transAxes,
                fontsize=4.5, va="bottom", ha="left", color="#666666", style="italic")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(0.6)

    fig.suptitle(f"Cascading Randomization — {display_name} architecture", fontweight="bold", fontsize=8, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = f"results/synthetic/cascading_randomization_figures/{out_slug}_183mm"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.pdf / .png  (width={TOTAL_WIDTH_IN:.3f} in = {TOTAL_WIDTH_MM} mm, {n} panels)")


def main():
    for display_name, (src_dir, out_slug) in ARCHITECTURES.items():
        make_figure(display_name, src_dir, out_slug)


if __name__ == "__main__":
    main()
