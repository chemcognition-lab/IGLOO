#!/usr/bin/env python3
"""
Presentation-ready ("simple") metric-vs-model-performance figures.

Same underlying data and metric code as ``plot_ndcg_vs_model_performance.py``,
but rendered for slides: short titles, short axis labels, concise legends, and
no statistics caption block. These are the figures used in the noise-peak
investigation deck (see ``docs/REPRODUCTION.md`` for the slide -> mode mapping).

Modes
-----
interp-panels   One panel per interpolation path (linear / slerp / nearest_neighbor),
                single metric. Reproduces "Attempt 1: reframe off noise level".
metric-grid     One panel per metric, with the shuffled-label random baseline
                overlaid; also writes a combined 2x2 grid.
                Reproduces "Attempt 2: random baseline".
gaussian        Single-metric panel plus the large-Gaussian-noise cell
                (sigma=20) as an extra point. Reproduces "Attempt 3".
regression      Magnitude (MAE on z-scored vectors) metrics.
                Reproduces "Attempt 6: magnitude-based metrics".

The reduced model ("Attempt 5") is any of the above with
``--results-root results/synthetic_reduced_model``.

Usage
-----
    python scripts/plot_simple_figures.py --mode interp-panels
    python scripts/plot_simple_figures.py --mode metric-grid
    python scripts/plot_simple_figures.py --mode gaussian
    python scripts/plot_simple_figures.py --mode regression
    python scripts/plot_simple_figures.py --mode metric-grid \
        --results-root results/synthetic_reduced_model --label "reduced model"

Runtime note: each mode re-reads every per-cell integrated-gradients CSV
(60 sweep cells + 5 baseline cells per interpolation path, ~81k rows each), so
expect roughly 10 minutes for a single path and ~15-20 for all three. Run it in
the background or under Slurm rather than interactively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
if __name__ == "__main__":
    # Headless only when used as a CLI; importing this module into a notebook
    # must not clobber an already-configured inline backend.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.ig_metrics import NEAR_ZERO_THRESHOLD, compute_metrics_for_dir  # noqa: E402
from src.igloo.plot import FIG_BG, INTERP_LABELS  # noqa: E402
from scripts.plot_ndcg_vs_model_performance import (  # noqa: E402
    collect_metrics,
    collect_random_baseline,
    _ci95,
)

# Short, slide-friendly names. Keys match ig_metrics / plot_ndcg_vs_model_performance.
SHORT_LABELS = {
    "ndcg":                       "NDCG",
    "auroc":                      "AUROC",
    "ap_graded":                  "AP@R",
    "pv_ndcg_mean":               "Per-variant NDCG",
    "sign_acc":                   "Sign accuracy",
    "cat_sep":                    "Category separation",
    "mutation_ndcg_mean_per_seq": "Mutation NDCG",
    "ig_gt_mae_z":                "MAE (z-scored, global)",
    "mutation_mae_z":             "MAE (z-scored, mutation-level)",
}

GRID_METRICS = ["ndcg", "sign_acc", "cat_sep", "mutation_ndcg_mean_per_seq"]
REGRESSION_METRICS = ["ig_gt_mae_z", "mutation_mae_z"]

NAVY = "#1D3557"
RED = "#E63946"
DARK_RED = "#7f1d1d"
TEAL = "#2A9D8F"


def _aggregate(sub, y_metric):
    """Per-noise-level mean and 95% CI across seeds."""
    noise_levels = sorted(sub["noise"].unique())
    return (
        sub.groupby("noise")
        .agg(
            x_mean=("x_value", "mean"), x_ci=("x_value", lambda v: _ci95(v.values)),
            y_mean=(y_metric, "mean"), y_ci=(y_metric, lambda v: _ci95(v.values)),
        )
        .reindex(noise_levels)
        .reset_index()
    ), noise_levels


def _draw_panel(ax, fig, sub, agg, noise_levels, y_metric, *, annotate_noise=True,
                compact=False):
    """Scatter of individual runs + per-noise-level mean/CI trace."""
    sca = ax.scatter(
        sub["x_value"], sub[y_metric], c=sub["noise"], cmap=plt.get_cmap("viridis"),
        norm=plt.Normalize(vmin=min(noise_levels), vmax=max(noise_levels)),
        s=45 if compact else 55, alpha=0.55, edgecolors="white", linewidths=0.6,
        zorder=2, label="individual runs",
    )
    cbar = fig.colorbar(sca, ax=ax, pad=0.02)
    cbar.set_label("Noise level", fontsize=8.5 if compact else 9.5)

    ax.errorbar(
        agg["x_mean"], agg["y_mean"], xerr=agg["x_ci"], yerr=agg["y_ci"],
        fmt="D-", color=NAVY, ecolor=NAVY, elinewidth=1.2 if compact else 1.4,
        capsize=3.5 if compact else 4, ms=7 if compact else 9, mfc="white",
        mew=1.6 if compact else 1.8, lw=1.2 if compact else 1.4, alpha=0.95,
        zorder=3, label="per-noise mean ± 95% CI",
    )
    if annotate_noise:
        for _, row in agg.iterrows():
            ax.annotate(
                f"{row['noise']:g}", (row["x_mean"], row["y_mean"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8 if compact else 8.5,
                color=NAVY, fontweight="bold",
            )


def _draw_baseline(ax, baseline_df, y_metric, *, compact=False):
    """Shuffled-label random baseline: individual seeds, mean, and its CI band."""
    sub = baseline_df.dropna(subset=[y_metric]) if baseline_df is not None else None
    if sub is None or sub.empty:
        return
    bx, by = sub["x_value"].values, sub[y_metric].values
    bx_mean, bx_ci = float(bx.mean()), _ci95(bx)
    by_mean, by_ci = float(by.mean()), _ci95(by)

    ax.axvspan(bx_mean - bx_ci, bx_mean + bx_ci, color=RED, alpha=0.08, zorder=0)
    ax.axvline(bx_mean, color=RED, ls="--", lw=1.2 if compact else 1.3, alpha=0.7, zorder=1)
    ax.scatter(bx, by, marker="x", s=38 if compact else 45, color=RED, alpha=0.6,
               zorder=2, label="random baseline")
    ax.errorbar(
        [bx_mean], [by_mean], xerr=[bx_ci], yerr=[by_ci],
        fmt="*", color=RED, ecolor=RED, elinewidth=1.4 if compact else 1.6,
        capsize=4 if compact else 5, ms=16 if compact else 20, mec=DARK_RED, mew=1.0,
        zorder=4, label="random baseline mean",
    )


def _finish(ax, title, ylabel, *, compact=False, legend=True):
    ax.set_xlabel("Model performance (Spearman ρ)", fontsize=10 if compact else 11)
    ax.set_ylabel(ylabel, fontsize=11 if compact else 12)
    ax.set_title(title, fontsize=12.5 if compact else 13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(fontsize=7.5 if compact else 9, loc="best", framealpha=0.6)


def _save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["interp-panels", "metric-grid", "gaussian", "regression"])
    p.add_argument("--results-root", default="results/synthetic",
                   help="Root holding {protein}_{fitness_model}/noise_*/seed_*/{interp}/. "
                        "Use results/synthetic_reduced_model for the reduced-model sweep.")
    p.add_argument("--out-dir", default="results/synthetic/figures/simple_version")
    p.add_argument("--protein", default="GB1")
    p.add_argument("--fitness-model", default="additive")
    p.add_argument("--interp", default="linear")
    p.add_argument("--metric", default="ndcg", help="Metric for interp-panels / gaussian modes.")
    p.add_argument("--label", default="", help="Extra title tag, e.g. 'reduced model'.")
    p.add_argument("--no-legend", action="store_true",
                   help="Drop the legend entirely (used for the 3-panel slide figure).")
    p.add_argument("--gaussian-cell",
                   default="gaussian_noise_std_20.0/seed_1",
                   help="Cell (relative to {protein}_{fitness_model}/) for the large-Gaussian point.")
    args = p.parse_args()

    results_root = (_PROJECT_ROOT / args.results_root).resolve()
    out_dir = (_PROJECT_ROOT / args.out_dir).resolve()
    tag = f" ({args.label})" if args.label else ""
    prefix = f"{args.protein}_{args.fitness_model}"

    interps = ["linear", "slerp", "nearest_neighbor"] if args.mode == "interp-panels" else [args.interp]
    if args.mode == "metric-grid":
        metrics = GRID_METRICS
    elif args.mode == "regression":
        metrics = REGRESSION_METRICS
    else:
        metrics = [args.metric]

    print(f"Collecting metrics from {results_root} (interps={interps}, metrics={metrics}) ...")
    df = collect_metrics(results_root, args.protein, args.fitness_model, interps,
                         NEAR_ZERO_THRESHOLD, 5, "model_test_spearman", metrics)
    print(f"  {len(df)} (noise, seed, interp) cells collected.")

    baseline_df = None
    if args.mode in {"metric-grid", "gaussian", "regression"}:
        print("Collecting shuffled-label random baseline ...")
        baseline_df = collect_random_baseline(results_root, args.protein, args.fitness_model,
                                              args.interp, NEAR_ZERO_THRESHOLD, 5,
                                              "model_test_spearman", metrics)
        print(f"  {len(baseline_df)} random-baseline cells collected.")

    # ---- interp-panels: one panel per interpolation path ------------------
    if args.mode == "interp-panels":
        fig, axes = plt.subplots(1, len(interps), figsize=(6.3 * len(interps), 5.2),
                                 facecolor=FIG_BG)
        for ax, interp in zip(axes, interps):
            ax.set_facecolor(FIG_BG)
            sub = df[df["interp"] == interp].dropna(subset=[args.metric]).copy()
            agg, noise_levels = _aggregate(sub, args.metric)
            _draw_panel(ax, fig, sub, agg, noise_levels, args.metric, compact=True)
            _finish(ax, f"{prefix.replace('_', ' ')}{tag} — {INTERP_LABELS.get(interp, interp)}",
                    SHORT_LABELS.get(args.metric, args.metric),
                    compact=True, legend=not args.no_legend)
        _save(fig, out_dir / f"{args.metric}_vs_spearman_by_interp_simple.png")
        return

    # ---- per-metric panels (+ combined grid for metric-grid) --------------
    for y_metric in metrics:
        sub = df[df["interp"] == args.interp].dropna(subset=[y_metric]).copy()
        agg, noise_levels = _aggregate(sub, y_metric)

        fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=FIG_BG)
        ax.set_facecolor(FIG_BG)
        _draw_panel(ax, fig, sub, agg, noise_levels, y_metric)
        _draw_baseline(ax, baseline_df, y_metric)

        if args.mode == "gaussian":
            gdir = results_root / prefix / args.gaussian_cell / args.interp
            m_g = compute_metrics_for_dir(gdir, protein=args.protein,
                                          fitness_model=args.fitness_model)
            if m_g is None:
                raise SystemExit(f"No usable Gaussian-noise cell at {gdir}. "
                                 "Generate it first: jobs/run_high_gaussian_noise_cell.sh")
            gx, gy = m_g["model_test_spearman"], m_g[y_metric]
            print(f"  Gaussian cell: spearman={gx:.4f}, {y_metric}={gy:.4f}")
            ax.scatter([gx], [gy], marker="P", s=260, color=TEAL, edgecolors="black",
                       linewidths=1.3, zorder=5, label="Gaussian noise (σ=20)")

        title = (f"{prefix.replace('_', ' ')}{tag} — "
                 f"{INTERP_LABELS.get(args.interp, args.interp)} — "
                 f"{SHORT_LABELS.get(y_metric, y_metric)}")
        _finish(ax, title, SHORT_LABELS.get(y_metric, y_metric), legend=not args.no_legend)

        suffix = "_with_gaussian_point" if args.mode == "gaussian" else ""
        stem = f"{prefix}_{y_metric}_vs_model_test_spearman_{args.interp}{suffix}_simple"
        _save(fig, out_dir / f"{stem}.png")

    # ---- combined 2x2 grid ------------------------------------------------
    if args.mode == "metric-grid":
        fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor=FIG_BG)
        for ax, y_metric in zip(axes.flatten(), metrics):
            ax.set_facecolor(FIG_BG)
            sub = df[df["interp"] == args.interp].dropna(subset=[y_metric]).copy()
            agg, noise_levels = _aggregate(sub, y_metric)
            _draw_panel(ax, fig, sub, agg, noise_levels, y_metric,
                        annotate_noise=False, compact=True)
            _draw_baseline(ax, baseline_df, y_metric, compact=True)
            _finish(ax, SHORT_LABELS.get(y_metric, y_metric),
                    SHORT_LABELS.get(y_metric, y_metric), compact=True)
        fig.suptitle(f"{prefix.replace('_', ' ')}{tag} — "
                     f"{INTERP_LABELS.get(args.interp, args.interp)}",
                     fontsize=15, fontweight="bold", y=1.0)
        _save(fig, out_dir / f"{prefix}_4metrics_vs_model_test_spearman_{args.interp}_combined.png")


if __name__ == "__main__":
    main()
