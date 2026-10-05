#!/usr/bin/env python3
"""
TOPSIS score across ankh_base intermediate layers (MeanPool architecture),
aggregated across landscapes by color -- same visual style as
make_poolfirst_topsis_figure.py (one subplot per protein, dark black
spines/text, grey provenance note, ±1 seed-std band), but x-axis = layer
depth instead of model performance.

Unlike the precomputed results/synthetic/metric_suite/layersweep_topsis_cells.csv
(one TOPSIS value per layer, aggregated from seed-mean metrics with no seed
spread left), this script recomputes TOPSIS per individual seed -- for each
(protein, landscape, seed), each layer's metrics are read at that layer's
best-noise operating point and TOPSIS-ranked against the other layers in the
same (protein, landscape, seed) group -- so the per-layer mean has a real
seed-to-seed std band, the same way make_poolfirst_topsis_figure.py gets its
bands across the seedsweep.

Also plots the model's actual test Spearman performance at each layer's
best-noise operating point as a dashed line in the same landscape color
(real data, not a fit).

Layer 0 is the raw (non-contextualized) token representation prior to any
transformer block -- NOT labeled "embedding" here, since that term is
reserved elsewhere in this project for the pLM's final pooled protein
embedding used for attribution.

Source: results/synthetic/meanpool_ankh_base_layersweep/L<layer>/<protein>_
<landscape>/noise_*/seed_*/linear/*_position_importance_summary.csv
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
    all_metrics, _model_perf, _topsis_fill_worst,
)

SRC = Path("results/synthetic/meanpool_ankh_base_layersweep")
LANDSCAPE_COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}
LANDSCAPE_MARKERS = {"additive": "o", "epistatic": "s", "nonlinear": "D"}


def _load_raw_records(protein, landscape):
    """layer -> noise -> seed -> (mag, sgn, model_spearman)."""
    out = {}
    for L in LAYER_SWEEP_LAYERS:
        combo_dir = SRC / f"L{L}" / f"{protein}_{landscape}"
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


def per_seed_topsis_cells(protein, landscape, metrics):
    """Returns a DataFrame: layer, seed, topsis, model_spearman -- TOPSIS
    computed within each seed's own cross-layer comparison (layer is picked
    at ITS OWN best-noise operating point, same convention as
    layer_sweep_best_cell_table, just not seed-averaged away before the
    TOPSIS step). model_spearman is the real model test performance at that
    same (layer, seed, best-noise) point, carried through unchanged."""
    raw = _load_raw_records(protein, landscape)
    if not raw:
        return pd.DataFrame(columns=["layer", "seed", "topsis", "model_spearman"])

    # per layer: pick the best noise level (mean model_spearman across seeds)
    layer_best_noise = {}
    for L, recs in raw.items():
        by_noise = {}
        for noise, seed, mag, sgn, mp in recs:
            by_noise.setdefault(noise, []).append(mp)
        noise_means = {n: np.nanmean(v) for n, v in by_noise.items() if np.isfinite(np.nanmean(v))}
        if noise_means:
            layer_best_noise[L] = max(noise_means, key=noise_means.get)

    # per (layer, seed) metric vector + model_spearman at that layer's best noise
    per_layer_seed_metrics = {}  # (layer, seed) -> {metric: value}
    per_layer_seed_perf = {}  # (layer, seed) -> model_spearman
    for L, recs in raw.items():
        best_noise = layer_best_noise.get(L)
        if best_noise is None:
            continue
        for noise, seed, mag, sgn, mp in recs:
            if noise != best_noise:
                continue
            met = all_metrics(mag, protein, signed=sgn)
            per_layer_seed_metrics[(L, seed)] = {m: met.get(m, np.nan) for m in metrics}
            per_layer_seed_perf[(L, seed)] = mp

    seeds = sorted({s for (_, s) in per_layer_seed_metrics})
    rows = []
    for seed in seeds:
        layers_here = [L for L in LAYER_SWEEP_LAYERS if (L, seed) in per_layer_seed_metrics]
        if len(layers_here) < 2:
            continue
        wide = pd.DataFrame(
            {m: [per_layer_seed_metrics[(L, seed)][m] for L in layers_here] for m in metrics},
            index=layers_here,
        )
        sub = _topsis_fill_worst(wide)
        if len(sub) < 2:
            continue
        X = sub.to_numpy(dtype=float)
        norm = np.sqrt((X ** 2).sum(axis=0))
        norm[norm == 0] = 1.0
        R = X / norm
        ideal_best = R.max(axis=0)
        ideal_worst = R.min(axis=0)
        s_plus = np.sqrt(((R - ideal_best) ** 2).sum(axis=1))
        s_minus = np.sqrt(((R - ideal_worst) ** 2).sum(axis=1))
        denom = s_plus + s_minus
        denom[denom == 0] = 1.0
        c = s_minus / denom
        for L, score in zip(sub.index, c):
            rows.append({"layer": L, "seed": seed, "topsis": float(score),
                        "model_spearman": per_layer_seed_perf.get((L, seed), np.nan)})
    return pd.DataFrame(rows)


def main():
    composite_metrics = BACKBONE_METRICS
    proteins = [p for p in PROTEINS if (SRC.exists() and any(
        (SRC / f"L{L}" / f"{p}_{l}").is_dir() for L in LAYER_SWEEP_LAYERS for l in LANDSCAPES))]

    fig, axes = plt.subplots(1, len(proteins), figsize=(6.5 * len(proteins), 8.0))
    if len(proteins) == 1:
        axes = [axes]

    for ax, protein in zip(axes, proteins):
        n_seeds_seen = set()

        for landscape in LANDSCAPES:
            cells = per_seed_topsis_cells(protein, landscape, composite_metrics)
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
                    markersize=7, linewidth=1.8, alpha=0.95, label=landscape.capitalize())

            # real model test Spearman performance at the same (layer, best-noise)
            # points, same color, dashed -- not a fit, actual per-layer performance
            perf = cells.dropna(subset=["model_spearman"]).groupby("layer")["model_spearman"].mean()
            perf = perf.reindex(agg.layer)
            ax.plot(agg.layer, perf.to_numpy(), "--", color=color, linewidth=1.6, alpha=0.75, zorder=1)

        ax.set_xticks(LAYER_SWEEP_LAYERS)
        ax.set_title(f"{protein} — ankh_base / MeanPool", fontweight="bold", fontsize=22, color="black")
        ax.set_xlabel("ankh_base layer depth", fontsize=18, color="black")
        ax.set_ylabel("Attribution Quality (TOPSIS)", fontsize=18, color="black")
        ax.tick_params(axis="both", labelsize=15, labelcolor="black", color="black")
        ax.set_ylim(-0.02, 1.05)
        legend = ax.legend(loc="lower right", fontsize=15, title="Landscape", title_fontsize=15)
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("black")
        legend.get_title().set_color("black")
        ax.add_artist(legend)
        style_handles = [
            plt.Line2D([0], [0], color="#444444", marker="o", linestyle="-", lw=1.8, ms=6,
                       label="TOPSIS (attribution quality)"),
            plt.Line2D([0], [0], color="#444444", linestyle="--", lw=1.6, label="Model test Spearman ρ"),
        ]
        style_legend = ax.legend(handles=style_handles, loc="upper left", fontsize=13.5, framealpha=0.9)
        style_legend.get_frame().set_edgecolor("black")
        for text in style_legend.get_texts():
            text.set_color("black")
        n_seeds = max(n_seeds_seen) if n_seeds_seen else 0
        ax.text(0.03, 0.03, f"up to {len(n_seeds_seen)} seeds/layer",
                transform=ax.transAxes, fontsize=13.5, va="bottom", ha="left", color="#666666", style="italic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.6)

    fig.suptitle("TOPSIS Score Across ankh_base Layer Depth — MeanPool architecture",
                 fontweight="bold", fontsize=23, color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "results/synthetic/metric_suite/layersweep_topsis_by_landscape"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
