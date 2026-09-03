#!/usr/bin/env python3
"""
NDCG vs. model performance (reframing the noise-level axis).

Motivation
----------
Figure 2A (Global NDCG vs. noise level) shows a non-intuitive intermediate
peak around noise ~0.3-0.5 for GB1 additive. "Noise level" is an arbitrary
knob (label_noise_std) whose meaning is hard to interpret directly, and at
a fixed noise level the *actual* difficulty of the fitted surrogate model
still varies run to run (seed variance in training). This script reframes
the x-axis as the surrogate model's own predictive performance on held-out
test data (Spearman rho between predicted and true fitness, computed by
ig_metrics.compute_metrics_for_dir) instead of the noise-level knob, to see
whether attribution quality (NDCG) tracks model performance more cleanly
than it tracks noise level.

Produces, for GB1 additive across the full 0.0-1.0 decile noise grid
(11 levels x 5 seeds):
  - A scatter of all (seed, noise) cells: x = model test performance,
    y = global NDCG, points colored by noise level.
  - Per-noise-level mean +/- 95% CI (across the 5 seeds) overlaid as larger
    markers, connected in noise-level order.
  - Pearson/Spearman correlation between model performance and NDCG,
    computed both on the raw (seed, noise) cells and on the per-noise-level
    means, printed on the figure and to stdout.

Usage
-----
    python scripts/plot_ndcg_vs_model_performance.py
    python scripts/plot_ndcg_vs_model_performance.py --interp linear slerp nearest_neighbor
    python scripts/plot_ndcg_vs_model_performance.py --x-metric model_test_r2
    # All four Figure-2 signal families (global/per-variant/category/mutation-level):
    python scripts/plot_ndcg_vs_model_performance.py --y-metric ndcg sign_acc cat_sep mutation_ndcg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
if __name__ == "__main__":
    # Only force the headless, file-only backend when run as a CLI script
    # (e.g. on Slurm with no display). Importing this module into an
    # already-configured interactive kernel (Jupyter/Colab, %matplotlib
    # inline) must NOT clobber that backend, or plt.show() silently stops
    # rendering anything -- text output still works, but no figure appears.
    matplotlib.use("Agg")
matplotlib.rcParams["savefig.transparent"] = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.ig_metrics import (  # noqa: E402
    discover_result_dirs_multiseed,
    compute_metrics_for_dir,
    NEAR_ZERO_THRESHOLD,
)
from src.igloo.plot import FIG_BG, INTERP_COLORS, INTERP_LABELS, INTERP_MARKERS  # noqa: E402

X_METRIC_LABELS = {
    "model_test_spearman": "Model test performance (Spearman ρ, predicted vs. true fitness)",
    "model_test_kendall":  "Model test performance (Kendall τ, predicted vs. true fitness)",
    "model_test_ndcg":     "Model test performance (NDCG, predicted vs. true fitness)",
    "model_test_r2":       "Model test performance (R², predicted vs. true fitness)",
}

# All 7 canonical metrics from method_selection.COMPOSITE_METRICS, grouped
# into the same 4 families as Figure 2 (see src/igloo/plot.py make_figure()).
# "mutation_ndcg_mean_per_seq" is a derived field (mean of the per-sequence
# mutation_ndcg_vals list) -- NOT the same as the raw "mutation_ndcg" scalar,
# which pools all sequences' mutations together before ranking. The violin
# plot in the original Figure 2D (and its per-noise mean, the black dot) is
# built from the per-sequence version, so that's what belongs here; see
# _extract_y_value() below for how it's computed.
Y_METRIC_LABELS = {
    "ndcg":         ("Global NDCG (↑ better)", "Global ranking"),
    "auroc":        ("AUROC, annotated vs background (↑ better)", "Global ranking"),
    "ap_graded":    ("AP@R, graded relevance (↑ better)", "Global ranking"),
    "pv_ndcg_mean": ("Per-variant NDCG, mean (↑ better)", "Per-variant"),
    "sign_acc":     ("Sign accuracy — fraction correct (↑ better)", "Per-variant"),
    "cat_sep":      ("Category separation ratio (↑ better)", "Category-level"),
    "mutation_ndcg_mean_per_seq": ("Mutation-level NDCG, per-sequence mean (↑ better)", "Mutation-level"),
}

# Regression-style companions to two of the 7 rank/direction-based metrics
# above: Pearson correlation instead of rank correlation, i.e. does
# attribution *magnitude* scale linearly with true importance, not just
# agree in relative order. Kept as a separate registry (not merged into
# ALL_METRIC_KEYS) so the original 7-metric sweep is untouched by default;
# pass these keys explicitly via --y-metric to plot them.
REGRESSION_METRIC_LABELS = {
    "ig_gt_pearson_r": ("Pearson r(|IG|, ground-truth importance) (↑ better)", "Global ranking (regression)"),
    "mutation_pearson_mean_per_seq": ("Mutation-level Pearson r(|IG|, LOO Δfitness), per-sequence mean (↑ better)", "Mutation-level (regression)"),
    # z-scored MAE/R²: true magnitude-agreement metrics (sensitive to scale
    # and slope, unlike Pearson which is scale/offset-invariant). R² here is
    # against the y=x identity line, not the best-fit regression line.
    "ig_gt_mae_z": ("MAE(z-scored |IG|, z-scored ground truth) (↓ better)", "Global ranking (regression)"),
    "ig_gt_r2_z":  ("R²(z-scored |IG| vs. z-scored ground truth, identity line) (↑ better)", "Global ranking (regression)"),
    "mutation_mae_z": ("Mutation-level MAE(z-scored |IG|, z-scored LOO Δfitness), pooled (↓ better)", "Mutation-level (regression)"),
    "mutation_r2_z":  ("Mutation-level R²(z-scored |IG| vs. z-scored LOO Δfitness, identity line), pooled (↑ better)", "Mutation-level (regression)"),
}
Y_METRIC_LABELS.update(REGRESSION_METRIC_LABELS)

ALL_METRIC_KEYS = [k for k in Y_METRIC_LABELS if k not in REGRESSION_METRIC_LABELS]
REGRESSION_METRIC_KEYS = list(REGRESSION_METRIC_LABELS)


def _extract_y_value(m: dict, ym: str) -> float:
    if ym == "mutation_ndcg_mean_per_seq":
        vals = m.get("mutation_ndcg_vals", [])
        return float(np.mean(vals)) if vals else float("nan")
    if ym == "ig_gt_pearson_r":
        return m.get("pearson_r", float("nan"))
    if ym == "mutation_pearson_mean_per_seq":
        vals = m.get("mutation_pearson_vals", [])
        return float(np.mean(vals)) if vals else float("nan")
    if ym == "ig_gt_mae_z":
        return m.get("mae_z", float("nan"))
    if ym == "ig_gt_r2_z":
        return m.get("r2_z", float("nan"))
    if ym in ("mutation_mae_z", "mutation_r2_z"):
        return m.get(ym, float("nan"))
    return m.get(ym, float("nan"))


def collect_metrics(
    results_root: Path, protein: str, fitness_model: str,
    interps: list[str], min_output_diff: float, min_count: int,
    x_metric: str, y_metrics: list[str],
) -> pd.DataFrame:
    """One row per (noise, seed, interp) cell, with x_metric plus every
    requested y_metrics column pulled from a single compute_metrics_for_dir
    call (avoids re-parsing the same CSV once per signal family)."""
    rows = []
    cells = discover_result_dirs_multiseed(
        results_root, proteins=[protein], fitness_models=[fitness_model],
    )
    for prot, fm, noise, seed, interp, path in cells:
        if interp not in interps:
            continue
        m = compute_metrics_for_dir(
            path, protein=prot, fitness_model=fm,
            min_output_diff=min_output_diff, min_count=min_count,
        )
        if m is None:
            print(f"  [warn] no usable data: {path}")
            continue
        row = {"noise": noise, "seed": seed, "interp": interp, "x_value": m.get(x_metric, float("nan"))}
        for ym in y_metrics:
            row[ym] = _extract_y_value(m, ym)
        rows.append(row)
    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.dropna(subset=["x_value"])
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} cells with missing {x_metric}.")
    return df


def collect_random_baseline(
    results_root: Path, protein: str, fitness_model: str, interp: str,
    min_output_diff: float, min_count: int, x_metric: str, y_metrics: list[str],
    subdir: str = "random_baseline",
) -> pd.DataFrame:
    """Metrics for the shuffled-label "complete random model" negative control.

    Expects results_root/{protein}_{fitness_model}/{subdir}/seed_*/{interp}/
    (see jobs/run_random_baseline_cell.sh — training.noise.shuffle_labels=true).
    """
    base = results_root / f"{protein}_{fitness_model}" / subdir
    rows = []
    if not base.exists():
        print(f"  [warn] no random-baseline directory found at {base}")
        return pd.DataFrame(rows)
    for seed_dir in sorted(base.glob("seed_*")):
        path = seed_dir / interp
        if not path.exists():
            continue
        m = compute_metrics_for_dir(
            path, protein=protein, fitness_model=fitness_model,
            min_output_diff=min_output_diff, min_count=min_count,
        )
        if m is None:
            print(f"  [warn] no usable data: {path}")
            continue
        row = {"seed": seed_dir.name, "interp": interp, "x_value": m.get(x_metric, float("nan"))}
        for ym in y_metrics:
            row[ym] = _extract_y_value(m, ym)
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["x_value"])


def _ci95(vals: np.ndarray) -> float:
    """Half-width of a 95% CI via Student-t, for small n (e.g. 5 seeds)."""
    n = len(vals)
    if n < 2:
        return 0.0
    se = vals.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return float(t_crit * se)


def _fit_attenuation_noise_level(noise: np.ndarray, x_mean: np.ndarray, target_x: float) -> "tuple[float, float, float] | None":
    """Fit rho(s) = rho0*sigma_eff/sqrt(sigma_eff^2+s^2) to (noise, model-perf) means,
    and solve for the noise level s at which the fit reaches target_x.

    Returns (sigma_eff, rho0, r_squared, s_target) or None if the fit/solve fails.
    """
    from scipy.optimize import curve_fit

    def atten(s, rho0, sigma_eff):
        return rho0 * sigma_eff / np.sqrt(sigma_eff ** 2 + s ** 2)

    try:
        popt, _ = curve_fit(atten, noise, x_mean, p0=[max(x_mean.max(), 0.1), 0.5], maxfev=10000)
    except Exception as e:
        print(f"  [warn] attenuation fit failed: {e}")
        return None
    rho0, sigma_eff = popt
    resid = x_mean - atten(noise, *popt)
    r2 = 1 - np.sum(resid ** 2) / np.sum((x_mean - x_mean.mean()) ** 2)
    if target_x <= 0 or target_x >= rho0:
        return rho0, sigma_eff, r2, float("nan")
    s_target = np.sqrt((rho0 * sigma_eff / target_x) ** 2 - sigma_eff ** 2)
    return rho0, sigma_eff, r2, float(s_target)


def plot_one_interp(df: pd.DataFrame, interp: str, x_metric: str, y_metric: str,
                     output_path: "Path | None", dpi: int,
                     baseline_df: "pd.DataFrame | None" = None,
                     show: bool = False) -> "plt.Figure | None":
    """Build the metric-vs-model-performance scatter for one interp method.

    output_path=None skips saving to disk (e.g. for inline notebook use);
    show=True calls plt.show() instead of plt.close(), and the Figure is
    returned either way so callers (notebooks) can inspect/re-render it.
    """
    sub = df[df["interp"] == interp].dropna(subset=[y_metric]).copy()
    if sub.empty:
        print(f"  [warn] no data for interp={interp}, y_metric={y_metric}, skipping figure.")
        return None

    y_label, family_label = Y_METRIC_LABELS.get(y_metric, (y_metric, y_metric))

    noise_levels = sorted(sub["noise"].unique())
    agg = (
        sub.groupby("noise")
        .agg(
            x_mean=("x_value", "mean"), x_ci=("x_value", lambda v: _ci95(v.values)),
            y_mean=(y_metric, "mean"), y_ci=(y_metric, lambda v: _ci95(v.values)),
            n=(y_metric, "size"),
        )
        .reindex(noise_levels)
        .reset_index()
    )

    # Correlations
    r_raw_p  = stats.pearsonr(sub["x_value"], sub[y_metric])
    r_raw_s  = stats.spearmanr(sub["x_value"], sub[y_metric])
    r_agg_p  = stats.pearsonr(agg["x_mean"], agg["y_mean"])
    r_agg_s  = stats.spearmanr(agg["x_mean"], agg["y_mean"])

    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(noise_levels), vmax=max(noise_levels))
    sca = ax.scatter(
        sub["x_value"], sub[y_metric], c=sub["noise"], cmap=cmap, norm=norm,
        s=55, alpha=0.55, edgecolors="white", linewidths=0.6, zorder=2,
        label=f"individual runs (n={len(sub)} = {sub['noise'].nunique()} noise × {sub['seed'].nunique()} seeds)",
    )
    cbar = fig.colorbar(sca, ax=ax, pad=0.02)
    cbar.set_label("Noise level (label_noise_std)", fontsize=10)

    ax.errorbar(
        agg["x_mean"], agg["y_mean"], xerr=agg["x_ci"], yerr=agg["y_ci"],
        fmt="D-", color="#1D3557", ecolor="#1D3557", elinewidth=1.4,
        capsize=4, ms=9, mfc="white", mew=1.8, lw=1.4, alpha=0.95, zorder=3,
        label="per-noise-level mean ± 95% CI (5 seeds)",
    )
    for _, row in agg.iterrows():
        ax.annotate(
            f"{row['noise']:g}", (row["x_mean"], row["y_mean"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8.5,
            color="#1D3557", fontweight="bold",
        )

    extrap_txt = ""
    baseline_sub = baseline_df.dropna(subset=[y_metric]) if baseline_df is not None else None
    if baseline_sub is not None and not baseline_sub.empty:
        bx = baseline_sub["x_value"].values
        by = baseline_sub[y_metric].values
        bx_mean, bx_ci = float(bx.mean()), _ci95(bx)
        by_mean, by_ci = float(by.mean()), _ci95(by)

        # Shaded band marking "random model performance" on the x-axis
        ax.axvspan(bx_mean - bx_ci, bx_mean + bx_ci, color="#E63946", alpha=0.08, zorder=0)
        ax.axvline(bx_mean, color="#E63946", ls="--", lw=1.3, alpha=0.7, zorder=1)

        ax.scatter(
            bx, by, marker="x", s=45, color="#E63946", alpha=0.6, zorder=2,
            label=f"random baseline, individual seeds (n={len(baseline_sub)})",
        )
        ax.errorbar(
            [bx_mean], [by_mean], xerr=[bx_ci], yerr=[by_ci],
            fmt="*", color="#E63946", ecolor="#E63946", elinewidth=1.6,
            capsize=5, ms=20, mec="#7f1d1d", mew=1.0, zorder=4,
            label="random baseline mean ± 95% CI (shuffled labels, 5 seeds)",
        )
        ax.annotate(
            "random\n(shuffled labels)", (bx_mean, by_mean),
            textcoords="offset points", xytext=(8, -22), fontsize=8.5,
            color="#7f1d1d", fontweight="bold",
        )

        # Target the *upper* edge of the baseline's CI (where the real curve
        # first enters plausibly-random territory), not the raw mean -- with
        # only 5 seeds the baseline mean itself can land at/below 0 by chance
        # (as it does here), which has no finite solution under the
        # attenuation model (it only approaches 0 asymptotically).
        target_x = max(bx_mean + bx_ci, 1e-3)
        fit = _fit_attenuation_noise_level(
            agg["noise"].values.astype(float), agg["x_mean"].values, target_x,
        )
        if fit is not None:
            rho0, sigma_eff, r2, s_target = fit
            noise_max_tested = agg["noise"].max()
            if np.isfinite(s_target):
                extrap_txt = (
                    f"Attenuation-model fit of model-perf vs. noise (R²={r2:.3f}): "
                    f"real model enters the random-baseline's 95% CI band "
                    f"(x≤{target_x:.3f}) at noise ≈ {s_target:.1f} "
                    f"(extrapolated; tested range only goes up to {noise_max_tested:g})"
                )
            else:
                extrap_txt = (
                    f"Attenuation-model fit (R²={r2:.3f}) does not reach the random-baseline "
                    f"CI band within a finite noise value from this data."
                )
            print(f"  [{interp}] {extrap_txt}")

    ax.set_xlabel(X_METRIC_LABELS.get(x_metric, x_metric), fontsize=11)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"GB1 additive — {INTERP_LABELS.get(interp, interp)} — {family_label}\n"
        f"{y_label} vs. model performance, reframed off noise level",
        fontsize=13, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="best", framealpha=0.6)

    txt = (
        f"raw (n={len(sub)}): Pearson r={r_raw_p.statistic:.3f} (p={r_raw_p.pvalue:.3g}), "
        f"Spearman ρ={r_raw_s.statistic:.3f} (p={r_raw_s.pvalue:.3g})\n"
        f"per-noise means (n={len(agg)}): Pearson r={r_agg_p.statistic:.3f} (p={r_agg_p.pvalue:.3g}), "
        f"Spearman ρ={r_agg_s.statistic:.3f} (p={r_agg_s.pvalue:.3g})"
    )
    if extrap_txt:
        txt += "\n" + extrap_txt
    ax.text(
        0.02, -0.17, txt, transform=ax.transAxes, fontsize=8.5,
        color="#444", va="top", wrap=True,
    )

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"  Saved: {output_path}")
    if show:
        plt.show()
    elif output_path is not None:
        plt.close(fig)

    print(f"\n  [{interp}] correlation(model_perf, {y_metric}):")
    print(f"    raw cells      : Pearson r={r_raw_p.statistic:.3f} (p={r_raw_p.pvalue:.3g}),  Spearman rho={r_raw_s.statistic:.3f} (p={r_raw_s.pvalue:.3g})")
    print(f"    per-noise means: Pearson r={r_agg_p.statistic:.3f} (p={r_agg_p.pvalue:.3g}),  Spearman rho={r_agg_s.statistic:.3f} (p={r_agg_s.pvalue:.3g})")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protein", default="GB1")
    parser.add_argument("--fitness-model", default="additive")
    parser.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results" / "synthetic"))
    parser.add_argument("--output-dir", default=str(_PROJECT_ROOT / "results" / "synthetic" / "figures"))
    parser.add_argument("--interp", nargs="+", default=["linear"],
                         choices=["linear", "slerp", "nearest_neighbor"])
    parser.add_argument("--x-metric", default="model_test_spearman",
                         choices=list(X_METRIC_LABELS))
    parser.add_argument("--y-metric", nargs="+", default=["ndcg"],
                         choices=list(Y_METRIC_LABELS),
                         help="One or more signal-family metrics to plot as y-axis "
                              "(default: ndcg = global ranking, matching Figure 2A).")
    parser.add_argument("--min-output-diff", type=float, default=NEAR_ZERO_THRESHOLD)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--csv-out", default=None,
                         help="Optional path to dump the raw per-cell metrics table as CSV.")
    parser.add_argument("--no-random-baseline", action="store_true", default=False,
                         help="Skip the shuffled-label random-model baseline overlay.")
    parser.add_argument("--random-baseline-subdir", default="random_baseline",
                         help="Subdir under results_root/{protein}_{fitness_model}/ holding "
                              "seed_*/{interp}/ random-baseline runs.")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print(f"Collecting metrics for {args.protein} {args.fitness_model} "
          f"(interp={args.interp}, x_metric={args.x_metric}, y_metrics={args.y_metric}) "
          f"from {results_root} ...")
    df = collect_metrics(
        results_root, args.protein, args.fitness_model, args.interp,
        args.min_output_diff, args.min_count, args.x_metric, args.y_metric,
    )
    print(f"Collected {len(df)} (noise, seed, interp) cells across "
          f"{df['noise'].nunique()} noise levels x {df['seed'].nunique()} seeds "
          f"x {df['interp'].nunique()} interp method(s).")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  Saved raw metrics table -> {csv_path}")

    for interp in args.interp:
        baseline_df = None
        if not args.no_random_baseline:
            baseline_df = collect_random_baseline(
                results_root, args.protein, args.fitness_model, interp,
                args.min_output_diff, args.min_count, args.x_metric, args.y_metric,
                subdir=args.random_baseline_subdir,
            )
            print(f"  Random baseline ({interp}): {len(baseline_df)} seed(s) found.")
        for y_metric in args.y_metric:
            out_path = output_dir / f"{args.protein}_{args.fitness_model}_{y_metric}_vs_{args.x_metric}_{interp}.png"
            plot_one_interp(df, interp, args.x_metric, y_metric, out_path, args.dpi, baseline_df=baseline_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
