#!/usr/bin/env python3
"""
Composite 2-panel figure, one per protein (GB1, T4Lysozyme, p53), both
panels the SAME architecture -- MeanPool -- so they're directly comparable
(the two source figures this combines were originally built for two
DIFFERENT architectures: layersweep_topsis_by_landscape.png is MeanPool-only
(the only architecture the ankh_base layer-depth sweep was ever run for),
while poolfirst_ensemble_variance_by_landscape.png was Pool First (IG).
Panel B here is a NEW MeanPool-specific rebuild of the ensemble-variance
analysis, not a reuse of the Pool First one).

  Panel A: TOPSIS across ankh_base layer depth, MeanPool
           (make_layersweep_topsis_by_landscape_figure.py's per-protein panel)
  Panel B: Ensemble-size variance reduction, MeanPool at esmc_300m
           (same bootstrap-with-replacement methodology as
           make_poolfirst_ensemble_variance_figure.py, but against MeanPool's
           3-seed newarch_perres_mean checkpoints instead of Pool First's
           5-seed seedsweep -- so K only goes 1..3, not 1..5)

Panel A and Panel B deliberately use different backbones (ankh_base for the
layer sweep, esmc_300m for the ensemble analysis) because that's simply
where each experiment's raw data lives -- the thing that had to match, and
now does, is the ARCHITECTURE (MeanPool in both), not the backbone.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metric_suite import (  # noqa: E402
    PROTEINS, LANDSCAPES, LAYER_SWEEP_LAYERS, BACKBONE_METRICS,
    all_metrics, _model_perf, _topsis_fill_worst, topsis_score,
)

LANDSCAPE_COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}
LANDSCAPE_MARKERS = {"additive": "o", "epistatic": "s", "nonlinear": "D"}

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, NOTE_FS = 15, 13, 11, 10.5, 9.5

# ---------------------------------------------------------------------
# Panel A: layer-depth sweep (ankh_base, MeanPool) -- ported from
# make_layersweep_topsis_by_landscape_figure.py, one protein at a time.
# ---------------------------------------------------------------------
LAYERSWEEP_SRC = Path("results/synthetic/meanpool_ankh_base_layersweep")


def _load_layersweep_records(protein, landscape):
    out = {}
    for L in LAYER_SWEEP_LAYERS:
        combo_dir = LAYERSWEEP_SRC / f"L{L}" / f"{protein}_{landscape}"
        if not combo_dir.is_dir():
            continue
        recs = []
        for nd in sorted(combo_dir.glob("noise_*")):
            noise = float(nd.name.split("_", 1)[1])
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
                recs.append((noise, seed, mag, sgn, mp))
        if recs:
            out[L] = recs
    return out


def _layersweep_per_seed_topsis(protein, landscape):
    raw = _load_layersweep_records(protein, landscape)
    if not raw:
        return pd.DataFrame(columns=["layer", "seed", "topsis", "model_spearman"])

    layer_best_noise = {}
    for L, recs in raw.items():
        by_noise = {}
        for noise, seed, mag, sgn, mp in recs:
            by_noise.setdefault(noise, []).append(mp)
        noise_means = {n: np.nanmean(v) for n, v in by_noise.items() if np.isfinite(np.nanmean(v))}
        if noise_means:
            layer_best_noise[L] = max(noise_means, key=noise_means.get)

    per_layer_seed_metrics, per_layer_seed_perf = {}, {}
    for L, recs in raw.items():
        best_noise = layer_best_noise.get(L)
        if best_noise is None:
            continue
        for noise, seed, mag, sgn, mp in recs:
            if noise != best_noise:
                continue
            met = all_metrics(mag, protein, signed=sgn)
            per_layer_seed_metrics[(L, seed)] = {m: met.get(m, np.nan) for m in BACKBONE_METRICS}
            per_layer_seed_perf[(L, seed)] = mp

    seeds = sorted({s for (_, s) in per_layer_seed_metrics})
    rows = []
    for seed in seeds:
        layers_here = [L for L in LAYER_SWEEP_LAYERS if (L, seed) in per_layer_seed_metrics]
        if len(layers_here) < 2:
            continue
        wide = pd.DataFrame(
            {m: [per_layer_seed_metrics[(L, seed)][m] for L in layers_here] for m in BACKBONE_METRICS},
            index=layers_here,
        )
        sub = _topsis_fill_worst(wide)
        if len(sub) < 2:
            continue
        X = sub.to_numpy(dtype=float)
        norm = np.sqrt((X ** 2).sum(axis=0))
        norm[norm == 0] = 1.0
        R = X / norm
        ideal_best, ideal_worst = R.max(axis=0), R.min(axis=0)
        s_plus = np.sqrt(((R - ideal_best) ** 2).sum(axis=1))
        s_minus = np.sqrt(((R - ideal_worst) ** 2).sum(axis=1))
        denom = s_plus + s_minus
        denom[denom == 0] = 1.0
        c = s_minus / denom
        for L, score in zip(sub.index, c):
            rows.append({"layer": L, "seed": seed, "topsis": float(score),
                        "model_spearman": per_layer_seed_perf.get((L, seed), np.nan)})
    return pd.DataFrame(rows)


def panel_a_layersweep(ax, protein):
    has_data = any((LAYERSWEEP_SRC / f"L{L}" / f"{protein}_{l}").is_dir()
                   for L in LAYER_SWEEP_LAYERS for l in LANDSCAPES)
    if not has_data:
        ax.text(0.5, 0.5, "no layer-sweep data\nyet for this protein", transform=ax.transAxes,
                fontsize=12, color="#999999", ha="center", va="center", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return

    n_seeds_seen = set()
    for landscape in LANDSCAPES:
        cells = _layersweep_per_seed_topsis(protein, landscape)
        if cells.empty:
            continue
        n_seeds_seen.update(cells.seed.unique())
        agg = cells.groupby("layer").agg(mean=("topsis", "mean"), std=("topsis", "std")).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        color = LANDSCAPE_COLORS[landscape]
        marker = LANDSCAPE_MARKERS[landscape]
        ax.fill_between(agg.layer, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=color, alpha=0.2, linewidth=0)
        ax.plot(agg.layer, agg["mean"], f"{marker}-", color=color,
                markersize=5, linewidth=1.3, alpha=0.95, label=landscape.capitalize())
        perf = cells.dropna(subset=["model_spearman"]).groupby("layer")["model_spearman"].mean().reindex(agg.layer)
        ax.plot(agg.layer, perf.to_numpy(), "--", color=color, linewidth=1.3, alpha=0.75, zorder=1)

    ax.set_xticks(LAYER_SWEEP_LAYERS)
    ax.set_title("TOPSIS Across ankh_base Layer Depth", fontweight="bold", fontsize=TITLE_FS, color="black", pad=14)
    ax.set_xlabel("ankh_base layer depth", fontsize=LABEL_FS, color="black")
    ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=LABEL_FS, color="black")
    ax.tick_params(axis="both", labelsize=TICK_FS, labelcolor="black", color="black")
    ax.set_ylim(-0.02, 1.05)
    legend = ax.legend(loc="lower right", fontsize=LEGEND_FS, title="Landscape", title_fontsize=LEGEND_FS)
    legend.get_frame().set_edgecolor("black")
    for text in legend.get_texts():
        text.set_color("black")
    legend.get_title().set_color("black")
    ax.add_artist(legend)
    style_handles = [
        plt.Line2D([0], [0], color="#444444", marker="o", linestyle="-", lw=1.3, ms=5, label="TOPSIS"),
        plt.Line2D([0], [0], color="#444444", linestyle="--", lw=1.3, label="Model test Spearman ρ"),
    ]
    style_legend = ax.legend(handles=style_handles, loc="upper left", fontsize=9, framealpha=0.9)
    style_legend.get_frame().set_edgecolor("black")
    for text in style_legend.get_texts():
        text.set_color("black")
    ax.text(0.03, 0.03, f"{protein}, ankh_base, up to {len(n_seeds_seen)} seeds/layer",
            transform=ax.transAxes, fontsize=NOTE_FS, va="bottom", ha="left", color="#666666", style="italic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.2)


# ---------------------------------------------------------------------
# Panel B: ensemble-size variance reduction, MeanPool at esmc_300m.
# Same bootstrap-with-replacement methodology as
# make_poolfirst_ensemble_variance_figure.py, just pointed at MeanPool's
# MeanPool checkpoints. K goes as high as however many seeds actually exist
# at that (protein, landscape)'s best-noise point -- 10 for GB1 (extra-seed
# job 23146638), still 3 for T4Lysozyme/p53 until they get the same.
# ---------------------------------------------------------------------
ENSEMBLE_SRC = Path("results/synthetic/newarch_perres_mean")
ENSEMBLE_SEEDS = list(range(1, 11))  # up to 10 seeds where the extra-seed job ran (GB1 so far)
B_DRAWS = 300


def _load_ensemble_seed_vectors(protein, landscape):
    combo_dir = ENSEMBLE_SRC / f"{protein}_{landscape}"
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


def _ensemble_rows(protein, landscape):
    by_noise = _load_ensemble_seed_vectors(protein, landscape)
    if not by_noise:
        return []
    noise_means = {n: np.nanmean([v[2] for v in seeds.values()]) for n, seeds in by_noise.items()}
    noise_means = {n: v for n, v in noise_means.items() if np.isfinite(v)}
    if not noise_means:
        return []
    best_noise = max(noise_means, key=noise_means.get)
    seeds = by_noise[best_noise]
    available = sorted(s for s in ENSEMBLE_SEEDS if s in seeds)
    if not available:
        return []
    k_values = list(range(1, len(available) + 1))  # as many K's as seeds actually available

    rng = np.random.default_rng(hash((protein, landscape, "meanpool")) & 0xFFFFFFFF)
    rows = []
    for K in k_values:
        for _ in range(B_DRAWS):
            combo = rng.choice(available, size=K, replace=True)
            mags = np.mean([seeds[s][0] for s in combo], axis=0)
            sgns = np.mean([seeds[s][1] for s in combo], axis=0)
            met = all_metrics(mags, protein, signed=sgns)
            row = {"protein": protein, "landscape": landscape, "K": K}
            row.update({m: met.get(m, np.nan) for m in BACKBONE_METRICS})
            rows.append(row)
    return rows


def panel_b_ensemble_variance(ax, protein):
    all_rows = []
    for landscape in LANDSCAPES:
        all_rows.extend(_ensemble_rows(protein, landscape))
    df = pd.DataFrame(all_rows)
    if df.empty:
        ax.text(0.5, 0.5, "no MeanPool seedsweep\ndata yet for this protein", transform=ax.transAxes,
                fontsize=12, color="#999999", ha="center", va="center", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return
    df["topsis"] = topsis_score(df, BACKBONE_METRICS, group_cols=("protein", "landscape"))
    k_values = sorted(df.K.unique())

    for landscape in LANDSCAPES:
        g = df[df.landscape == landscape]
        if g.empty:
            continue
        color = LANDSCAPE_COLORS[landscape]
        std_by_k = g.groupby("K")["topsis"].std().reindex(k_values)
        ax.plot(k_values, std_by_k.to_numpy(), "o-", color=color, markersize=5,
                linewidth=1.3, alpha=0.95, label=landscape.capitalize(), zorder=3)
        std1 = std_by_k.loc[1]
        if np.isfinite(std1):
            ref = std1 / np.sqrt(np.array(k_values, dtype=float))
            ax.plot(k_values, ref, "--", color=color, linewidth=1.1, alpha=0.55, zorder=1)

    ax.set_xticks(k_values)
    ax.set_title("Ensemble-Size Variance Reduction", fontweight="bold", fontsize=TITLE_FS, color="black", pad=14)
    ax.set_xlabel("Number of seeds averaged into ensemble (K)", fontsize=LABEL_FS, color="black")
    ax.set_ylabel("SD of TOPSIS", fontsize=LABEL_FS, color="black")
    ax.tick_params(axis="both", labelsize=TICK_FS, labelcolor="black", color="black")
    ax.set_ylim(bottom=0)
    legend = ax.legend(loc="upper right", fontsize=LEGEND_FS, title="Landscape", title_fontsize=LEGEND_FS)
    legend.get_frame().set_edgecolor("black")
    for text in legend.get_texts():
        text.set_color("black")
    legend.get_title().set_color("black")
    ax.add_artist(legend)
    style_handles = [
        plt.Line2D([0], [0], color="#444444", marker="o", linestyle="-", lw=1.3, ms=5, label="Observed std. dev."),
        plt.Line2D([0], [0], color="#444444", linestyle="--", lw=1.1, label="Theoretical 1/√K (CLT)"),
    ]
    style_legend = ax.legend(handles=style_handles, loc="upper left", fontsize=9, framealpha=0.9)
    style_legend.get_frame().set_edgecolor("black")
    for text in style_legend.get_texts():
        text.set_color("black")
    ax.text(0.03, 0.03, f"{protein}, ESM-C 300M, {B_DRAWS} bootstrap draws/K (with replacement)",
            transform=ax.transAxes, fontsize=NOTE_FS, va="bottom", ha="left", color="#666666", style="italic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.2)


def main():
    out_dir = Path("results/synthetic/metric_suite")
    for protein in PROTEINS:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.0, 7.6))
        panel_a_layersweep(axA, protein)
        panel_b_ensemble_variance(axB, protein)
        axA.text(-0.14, 1.14, "A", transform=axA.transAxes, fontsize=19,
                 fontweight="bold", color="black", va="top", ha="left")
        axB.text(-0.14, 1.14, "B", transform=axB.transAxes, fontsize=19,
                 fontweight="bold", color="black", va="top", ha="left")
        fig.suptitle(f"MeanPool Architecture Summary — {protein}", fontweight="bold", fontsize=18, color="black")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out = out_dir / f"meanpool_layersweep_ensemble_{protein}"
        fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
