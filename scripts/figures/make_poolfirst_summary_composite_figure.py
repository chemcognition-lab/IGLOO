#!/usr/bin/env python3
"""
Composite 3-panel summary figure for the Pool First (IG) architecture, one
figure per protein (GB1, T4Lysozyme, p53 -- p53 panels render as "pending"
placeholders until its esmc_300m data / cascading-randomization data exist).

  Panel A: TOPSIS vs. model performance by landscape
           (make_poolfirst_topsis_figure.py's IG panel)
  Panel B: Cascading randomization by landscape, with the pooling-layer band
           (make_cascading_by_landscape_figure.py's Pool First panel)
  Panel C: Architecture comparison scatter -- Pool First vs. every other
           architecture (make_architecture_comparison_scatter.py's panel)

Each panel reproduces its source script's exact data/methodology for a
single protein; see those scripts' docstrings for the full methodology
notes (best-noise-operating-point convention, seed exclusions, etc.).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import topsis_score, BACKBONE_METRICS, PROTEINS, LANDSCAPES  # noqa: E402
from make_cascading_by_landscape_figure import (  # noqa: E402
    LANDSCAPE_COLORS, LANDSCAPE_MARKERS, load_all as load_cascading, _mark_pooling_step,
)

PRINT_PROTEINS = ["GB1", "T4Lysozyme", "p53"]
CASCADE_SRC_DIR = "results/synthetic/cascading_randomization_baseline"

METHOD_DISPLAY = {
    "IG": "Pool First (baseline)",
    "PreMLP": "Residual Pre-MLP",
    "MeanPool": "Per-residue + Mean Pool",
    "AttnPool": "Per-residue + Attn Pool",
    "OneHot": "One-Hot (baseline)",
}
METHOD_COLORS = {
    "IG": "#E69F00", "PreMLP": "#009E73", "MeanPool": "#0072B2",
    "AttnPool": "#CC79A7", "OneHot": "#999999",
}
METHOD_MARKERS = {"IG": "o", "PreMLP": "s", "MeanPool": "D", "AttnPool": "^", "OneHot": "X"}

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, NOTE_FS, LETTER_FS = 15, 13, 11, 10.5, 9.5, 20


def _panel_letter(ax, letter):
    ax.text(-0.14, 1.14, letter, transform=ax.transAxes, fontsize=LETTER_FS,
            fontweight="bold", color="black", va="top", ha="left")


def _empty_panel(ax, letter, title, msg):
    _panel_letter(ax, letter)
    ax.set_title(title, fontweight="bold", fontsize=TITLE_FS, color="black")
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, fontsize=12, color="#999999",
            ha="center", va="center", style="italic", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_a_topsis(ax, df_all, protein):
    sweep = df_all[(df_all.kind == "sweep") & (df_all.method == "IG") & (df_all.seed != 0)]
    p_df = sweep[sweep.protein == protein]
    if p_df.empty:
        _empty_panel(ax, "A", "TOPSIS vs. Model Performance", "no esmc_300m IG\nseedsweep data yet")
        return

    annot_lines = []
    for landscape in LANDSCAPES:
        g = p_df[p_df.landscape == landscape]
        if g.empty:
            continue
        agg = g.groupby("noise").agg(
            rho_mean=("model_spearman", "mean"),
            topsis_mean=("topsis_score", "mean"), topsis_std=("topsis_score", "std"),
            n=("topsis_score", "count"),
        ).reset_index().sort_values("rho_mean")
        agg = agg[agg.n == 5]

        color = LANDSCAPE_COLORS[landscape]
        marker = LANDSCAPE_MARKERS[landscape]
        ax.plot(agg.rho_mean, agg.topsis_mean, f"{marker}-", color=color,
                markersize=5, linewidth=1.3, alpha=0.95, label=landscape.capitalize())
        ax.fill_between(agg.rho_mean, agg.topsis_mean - agg.topsis_std.fillna(0),
                         agg.topsis_mean + agg.topsis_std.fillna(0), color=color, alpha=0.2, linewidth=0)

        s = g.dropna(subset=["model_spearman", "topsis_score"])
        if s.model_spearman.nunique() > 1:
            r, pval = pearsonr(s.model_spearman, s.topsis_score)
            annot_lines.append(f"{landscape.capitalize()}: r={r:.2f}")

    _panel_letter(ax, "A")
    ax.set_title("TOPSIS vs. Model Performance", fontweight="bold", fontsize=TITLE_FS, color="black")
    ax.set_xlabel("Model Test Spearman ρ", fontsize=LABEL_FS, color="black")
    ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=LABEL_FS, color="black")
    ax.tick_params(axis="both", labelsize=TICK_FS, labelcolor="black", color="black")
    ax.set_ylim(-0.02, 1.02)
    legend = ax.legend(loc="lower right", fontsize=LEGEND_FS, title="Landscape", title_fontsize=LEGEND_FS)
    legend.get_frame().set_edgecolor("black")
    for text in legend.get_texts():
        text.set_color("black")
    legend.get_title().set_color("black")
    if annot_lines:
        ax.text(0.03, 0.97, "\n".join(annot_lines), transform=ax.transAxes,
                fontsize=NOTE_FS, va="top", ha="left", color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


def panel_b_cascading(ax, df_cascade, protein):
    if df_cascade is None or df_cascade[df_cascade.protein == protein].empty:
        _empty_panel(ax, "B", "Cascading Randomization", "no cascading-randomization\ndata yet for this protein")
        return
    p_df = df_cascade[df_cascade.protein == protein]
    levels = p_df[["level", "label"]].drop_duplicates().sort_values("level")

    for landscape in LANDSCAPES:
        g = p_df[p_df.landscape == landscape]
        if g.empty:
            continue
        agg = g.groupby("level").agg(
            mean=("spearman_vs_original", "mean"), std=("spearman_vs_original", "std"),
        ).reindex(levels.level).reset_index()
        color = LANDSCAPE_COLORS[landscape]
        marker = LANDSCAPE_MARKERS[landscape]
        ax.plot(agg.level, agg["mean"], f"{marker}-", color=color, markersize=5,
                linewidth=1.3, alpha=0.95, label=landscape.capitalize())
        ax.fill_between(agg.level, agg["mean"] - agg["std"].fillna(0),
                         agg["mean"] + agg["std"].fillna(0), color=color, alpha=0.2, linewidth=0)

    _mark_pooling_step(ax, levels, "Pool First (IG)")
    _panel_letter(ax, "B")
    ax.set_xticks(levels.level)
    ax.set_xticklabels(levels.label, rotation=45, ha="right", fontsize=9)
    ax.set_title("Cascading Randomization", fontweight="bold", fontsize=TITLE_FS, color="black", pad=20)
    ax.set_xlabel("Cascading randomization step", fontsize=LABEL_FS, color="black")
    ax.set_ylabel("Spearman ρ vs. original attribution", fontsize=LABEL_FS, color="black")
    ax.tick_params(axis="y", labelsize=TICK_FS, labelcolor="black", color="black")
    ax.set_ylim(-0.08, 1.08)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    legend = ax.legend(loc="lower left", fontsize=LEGEND_FS, title="Landscape", title_fontsize=LEGEND_FS)
    legend.get_frame().set_edgecolor("black")
    for text in legend.get_texts():
        text.set_color("black")
    legend.get_title().set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


def _best_noise_point(g):
    by_noise = g.groupby("noise")["model_spearman"].mean()
    if by_noise.empty:
        return None
    best_noise = by_noise.idxmax()
    sub = g[g.noise == best_noise]
    return sub["model_spearman"].mean(), sub["topsis_score"].mean()


def panel_c_architecture(ax, sweep_df, protein):
    p_df = sweep_df[sweep_df.protein == protein]
    if p_df.empty:
        _empty_panel(ax, "C", "Architecture Comparison", "no esmc_300m sweep\ndata yet for this protein")
        return

    any_plotted = False
    for method in METHOD_DISPLAY:
        m_df = p_df[p_df.method == method]
        if method == "IG":
            m_df = m_df[m_df.seed != 0]
        if m_df.empty:
            continue
        land_points = []
        for landscape in LANDSCAPES:
            g = m_df[m_df.landscape == landscape]
            if g.empty:
                continue
            pt = _best_noise_point(g)
            if pt is not None:
                land_points.append(pt)
        if not land_points:
            continue
        any_plotted = True
        xs = np.array([p[0] for p in land_points])
        ys = np.array([p[1] for p in land_points])
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        ax.errorbar(xs.mean(), ys.mean(), xerr=xs.std(), yerr=ys.std(),
                    fmt=marker, color=color, ecolor=color, elinewidth=1.1, capsize=3,
                    markersize=8, markeredgecolor="black", markeredgewidth=0.7,
                    label=METHOD_DISPLAY[method], zorder=3)

    if not any_plotted:
        _empty_panel(ax, "C", "Architecture Comparison", "no esmc_300m sweep\ndata yet for this protein")
        return

    _panel_letter(ax, "C")
    ax.set_title("Architecture Comparison", fontweight="bold", fontsize=TITLE_FS, color="black")
    ax.set_xlabel("Model Test Spearman ρ", fontsize=LABEL_FS, color="black")
    ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=LABEL_FS, color="black")
    ax.tick_params(axis="both", labelsize=TICK_FS, labelcolor="black", color="black")
    ax.set_ylim(-0.02, 1.02)
    legend = ax.legend(loc="lower right", fontsize=9, title="Architecture", title_fontsize=9)
    legend.get_frame().set_edgecolor("black")
    for text in legend.get_texts():
        text.set_color("black")
    legend.get_title().set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


def main():
    df_all = pd.read_csv("results/synthetic/metric_suite/metric_suite.csv")
    df = df_all[df_all.backbone == "esmc_300m"].copy()
    composite_metrics = [m for m in BACKBONE_METRICS if m in df.columns]
    df["topsis_score"] = topsis_score(df, composite_metrics)
    sweep = df[df.kind == "sweep"]

    df_cascade = load_cascading(CASCADE_SRC_DIR)

    out_dir = Path("results/synthetic/poolfirst_summary_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for protein in PRINT_PROTEINS:
        fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(19.5, 7.0))

        panel_a_topsis(axA, sweep, protein)
        panel_b_cascading(axB, df_cascade, protein)
        panel_c_architecture(axC, sweep, protein)

        fig.suptitle(f"Pool First (IG) Summary — {protein}", fontweight="bold", fontsize=19, color="black")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out = out_dir / f"poolfirst_summary_{protein}"
        fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
