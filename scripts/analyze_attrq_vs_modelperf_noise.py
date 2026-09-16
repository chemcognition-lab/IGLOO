#!/usr/bin/env python3
"""
Attribution quality vs model test performance, coloured by label noise.

Analogue of results/Shen_Results/one_hot_IGLOO.png, built from our own
synthetic noise-sweep runs:

    results/synthetic/{Protein}_{landscape}/noise_{level}/{interp}/
        *_run_log_*.txt                    -> model test Spearman / R2 / Pearson
        *_position_importance_summary.csv  -> per-position attribution magnitude

Ground-truth position importance: data/synthetic/{Protein}_ground_truth_weights.json

For every (protein, landscape, interp) we get one point per noise level
(7 levels: 0.0 .. 1.0; 1 seed each, so no CI bands / shuffled-label baseline
like the Shen figure).  x = model test Spearman, y = attribution metric,
colour = noise level.

Usage:
    python scripts/analyze_attrq_vs_modelperf_noise.py \
        [--root <dir with the *_{landscape} noise-sweep folders>] \
        [--out  <output dir>]
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import ndcg_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:
    pass

PROTEINS = ["GB1", "T4Lysozyme"]
LANDSCAPES = ["additive", "epistatic", "nonlinear"]
# The noise-sweep runs have linear/slerp/nearest_neighbor sub-folders, but for
# noise >= 0.1 the attribution CSVs are byte-identical across them (the interp
# method did not actually vary the output / results were copied).  We therefore
# take `linear` as canonical and treat the data as one attribution per
# (protein, landscape, noise).
CANONICAL_INTERP = "linear"

_METRIC_RE = {
    "model_spearman": re.compile(r"Spearman\s*[ρr]\s*=\s*(-?\d+\.\d+)"),
    "model_r2":       re.compile(r"R²\s*=\s*(-?\d+\.\d+)"),
    "model_pearson":  re.compile(r"Pearson\s*r\s*=\s*(-?\d+\.\d+)"),
}


def parse_model_perf(run_log: Path) -> dict:
    """Pull the test-set metrics from the 'Model Performance:' block of a run log."""
    txt = run_log.read_text(errors="ignore")
    # the last occurrence is the test-set block (CV summary uses different wording)
    out = {}
    for key, rx in _METRIC_RE.items():
        m = rx.findall(txt)
        out[key] = float(m[-1]) if m else np.nan
    return out


def attribution_metrics(attr: np.ndarray, w: np.ndarray) -> dict:
    """|attr| per position vs ground-truth importance vector w."""
    abs_a = np.abs(attr)
    annotated = w > (w.min() + 1e-9)  # sparse importance: most positions are background

    sp, _ = spearmanr(abs_a, w)
    kt, _ = kendalltau(abs_a, w)
    ndcg = ndcg_score(w[None, :], abs_a[None, :])
    if annotated.any() and not annotated.all():
        auroc = roc_auc_score(annotated, abs_a)
        ap = average_precision_score(annotated, abs_a)
    else:
        auroc = ap = np.nan
    return {"attr_spearman": sp, "attr_kendall": kt, "attr_ndcg": ndcg,
            "attr_auroc": auroc, "attr_ap": ap}


_MUT_RE = re.compile(r"^[A-Z](\d+)[A-Z]$")


def per_variant_metrics(ig_csv: Path, w: np.ndarray) -> dict:
    """Per-variant (aggregation-LAST) attribution quality.

    Scores each analysed variant's own |IG| vector against the ground-truth
    importance vector, then reports the mean / std / SEM across variants.  This
    keeps the model-specific signal that the aggregation-first
    position_importance_summary.csv averages away.

    Also computes a 'mutation-focused' score: for each variant, does |IG| put
    that variant's actually-mutated positions above its unmutated ones (AUROC)?
    """
    df = pd.read_csv(ig_csv, usecols=["sequence_name", "position",
                                      "integrated_gradient"])
    L = len(w)
    annotated = w > (w.min() + 1e-9)

    ndcgs, sps, mut_aurocs = [], [], []
    for name, g in df.groupby("sequence_name", sort=False):
        if "WT" in name or "Poly" in name or "Activity" in name:
            continue
        g = g.sort_values("position")
        a = np.abs(g["integrated_gradient"].to_numpy())
        if len(a) != L or a.sum() == 0:
            continue
        ndcgs.append(ndcg_score(w[None, :], a[None, :]))
        sp, _ = spearmanr(a, w)
        if np.isfinite(sp):
            sps.append(sp)

        # mutation-focused: mutated positions (1-indexed) parsed from the name
        muts = {int(m.group(1)) for tok in name.split(";")
                if (m := _MUT_RE.match(tok.strip().split("[")[0]))}
        mut_mask = np.zeros(L, dtype=bool)
        for p in muts:
            if 1 <= p <= L:
                mut_mask[p - 1] = True
        if 0 < mut_mask.sum() < L:
            mut_aurocs.append(roc_auc_score(mut_mask, a))

    def _stats(v, prefix):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                    f"{prefix}_sem": np.nan, f"{prefix}_n": 0}
        return {f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std()),
                f"{prefix}_sem": float(v.std() / np.sqrt(len(v))), f"{prefix}_n": len(v)}

    return {**_stats(ndcgs, "pv_ndcg"), **_stats(sps, "pv_spearman"),
            **_stats(mut_aurocs, "pv_mut_auroc")}


_GT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
_GT_CACHE: dict[str, np.ndarray] = {}


def load_gt(protein: str) -> np.ndarray:
    if protein not in _GT_CACHE:
        j = json.loads((_GT_DIR / f"{protein}_ground_truth_weights.json").read_text())
        _GT_CACHE[protein] = np.asarray(j["importance_vector"], dtype=float)
    return _GT_CACHE[protein]


# metric name -> which random-null family to use
_NULL_KIND = {
    "attr_ndcg": "ndcg", "pv_ndcg_mean": "ndcg",
    "attr_spearman": "spearman", "attr_kendall": "spearman", "pv_spearman_mean": "spearman",
    "attr_auroc": "auroc", "pv_mut_auroc_mean": "mut_auroc",
    "attr_ap": "ap",
}


_RB_CACHE: dict = {}


def random_baseline(w: np.ndarray, metric: str, n: int = 3000, seed: int = 0) -> dict:
    """Distribution of *metric* under random position rankings (permutation null).

    Returns {'mean', 'lo', 'hi'} (2.5/97.5 percentiles), or None if no null defined.
    """
    kind = _NULL_KIND.get(metric)
    if kind is None:
        return None
    ck = (id(w), kind)
    if ck in _RB_CACHE:
        return _RB_CACHE[ck]
    rng = np.random.default_rng(seed)
    L = len(w)
    ann = w > (w.min() + 1e-9)
    vals = []
    for _ in range(n):
        s = rng.random(L)
        if kind == "ndcg":
            vals.append(ndcg_score(w[None, :], s[None, :]))
        elif kind == "spearman":
            vals.append(spearmanr(s, w)[0])
        elif kind == "auroc":
            vals.append(roc_auc_score(ann, s))
        elif kind == "ap":
            vals.append(average_precision_score(ann, s))
        elif kind == "mut_auroc":
            m = np.zeros(L, dtype=bool)
            m[rng.choice(L, size=2, replace=False)] = True  # ~typical variant: 2 muts
            vals.append(roc_auc_score(m, s))
    v = np.asarray(vals, dtype=float)
    out = {"mean": float(np.nanmean(v)),
           "lo": float(np.nanpercentile(v, 2.5)),
           "hi": float(np.nanpercentile(v, 97.5))}
    _RB_CACHE[ck] = out
    return out


def draw_random_baseline(ax, w: np.ndarray, metric: str) -> None:
    rb = random_baseline(w, metric)
    if rb is None:
        return
    ax.axhspan(rb["lo"], rb["hi"], color="#d62728", alpha=0.10, zorder=0)
    ax.axhline(rb["mean"], color="#d62728", ls="--", lw=1.1, zorder=0)
    ax.text(0.985, rb["mean"], " random ranking (mean, 95%)", color="#d62728",
            fontsize=7.5, va="bottom", ha="right", transform=ax.get_yaxis_transform())


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for protein in PROTEINS:
        for landscape in LANDSCAPES:
            base = root / f"{protein}_{landscape}"
            if not base.is_dir():
                continue
            w = load_gt(protein)

            for noise_dir in sorted(base.glob("noise_*")):
                noise = float(noise_dir.name.split("_", 1)[1])
                leaf = noise_dir / CANONICAL_INTERP
                run_logs = list(leaf.glob("*_run_log_*.txt")) or \
                    list(noise_dir.glob("*/*_run_log_*.txt"))
                perf = parse_model_perf(sorted(run_logs)[-1]) if run_logs else {}

                summ = list(leaf.glob("*_position_importance_summary.csv"))
                if not summ:
                    continue
                s = pd.read_csv(summ[0]).sort_values("position")
                attr = s["importance_score"].to_numpy()
                if len(attr) != len(w):
                    print(f"  skip {leaf} (len {len(attr)} != gt {len(w)})")
                    continue

                pv = {}
                ig_csv = list(leaf.glob("*_integrated_gradients_results.csv"))
                if ig_csv:
                    pv = per_variant_metrics(ig_csv[0], w)

                rows.append({
                    "protein": protein, "landscape": landscape, "noise": noise,
                    **perf, **attribution_metrics(attr, w), **pv,
                })
    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, metric: str, label: str, out_dir: Path) -> None:
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow),
                             squeeze=False, layout="constrained")
    norm = plt.Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis")

    for ax, (protein, landscape) in zip(axes.flat, combos):
        g = df[(df.protein == protein) & (df.landscape == landscape)].sort_values("noise")
        draw_random_baseline(ax, load_gt(protein), metric)
        ax.plot(g["model_spearman"], g[metric], color="#bbb", lw=1.0, zorder=1)
        sc = ax.scatter(g["model_spearman"], g[metric], c=g["noise"],
                        cmap=cmap, norm=norm, s=90,
                        edgecolors="black", linewidths=0.5, zorder=2)
        for _, r_ in g.iterrows():
            ax.annotate(f"{r_['noise']:g}", (r_["model_spearman"], r_[metric]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points",
                        color="#444")
        ax.set_title(f"{protein} — {landscape}", fontsize=10, fontweight="bold")
        sub = g.dropna(subset=["model_spearman", metric])
        if len(sub) >= 3 and sub["model_spearman"].nunique() >= 2:
            r, p = pearsonr(sub["model_spearman"], sub[metric])
            rho, _ = spearmanr(sub["model_spearman"], sub[metric])
            ax.text(0.03, 0.03, f"within-panel:\nPearson r={r:+.2f} (p={p:.2g})\n"
                                f"Spearman ρ={rho:+.2f}",
                    transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))
        ax.set_xlabel("Model test performance (Spearman ρ)")
        ax.set_ylabel(label)
        ax.axhline(0, color="#ddd", lw=0.8, zorder=0)

    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Label noise (std)")

    pooled = df.dropna(subset=["model_spearman", metric])
    pr, pp = pearsonr(pooled["model_spearman"], pooled[metric])
    fig.suptitle(f"{label} vs model test performance, coloured by label noise\n"
                 f"pooled over protein/landscape/interp (n={len(pooled)}): "
                 f"Pearson r={pr:+.2f} (p={pp:.1e})", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"attrq_vs_modelperf_noise_{metric}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir.name}/attrq_vs_modelperf_noise_{metric}.png / .svg")


def plot_shen_style(df: pd.DataFrame, metric: str, label: str, out_dir: Path,
                    err: str | None = None) -> None:
    """Single panel per (protein, landscape), styled like results/Shen_Results/one_hot_IGLOO.png.

    If *err* names a column, draw it as a y-error bar (e.g. per-variant SEM /
    std -- a within-run spread band, not a seed band).
    """
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.2 * nrow),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.58, wspace=0.24, top=0.88, bottom=0.09,
                        left=0.07, right=0.88)
    norm = plt.Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis")

    for ax, (protein, landscape) in zip(axes.flat, combos):
        g = df[(df.protein == protein) & (df.landscape == landscape)].sort_values("noise")
        sub = g.dropna(subset=["model_spearman", metric])

        draw_random_baseline(ax, load_gt(protein), metric)
        ax.plot(g["model_spearman"], g[metric], color="#4c5c96", lw=1.4,
                alpha=0.7, zorder=1)
        if err and err in g:
            ax.errorbar(g["model_spearman"], g[metric], yerr=g[err], fmt="none",
                        ecolor="#888", elinewidth=1.0, capsize=3, zorder=2)
        ax.scatter(g["model_spearman"], g[metric], c=g["noise"], cmap=cmap,
                   norm=norm, s=120, edgecolors="black", linewidths=0.6, zorder=3)
        for _, r_ in g.iterrows():
            ax.annotate(f"{r_['noise']:g}", (r_["model_spearman"], r_[metric]),
                        fontsize=7.5, xytext=(5, 5), textcoords="offset points",
                        color="#333", zorder=4)

        ax.set_title(f"{protein} — {landscape}", fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Model test performance (Spearman ρ, pred vs true fitness)",
                      fontsize=8.5)
        ax.set_ylabel(f"{label} (↑ better)", fontsize=8.5)

        band = "err bar = per-variant SEM" if err else "1 seed/noise"
        rb = random_baseline(load_gt(protein), metric)
        rb_txt = (f"random null {rb['mean']:.2f} [{rb['lo']:.2f}, {rb['hi']:.2f}]"
                  if rb else "")
        if len(sub) >= 3 and sub["model_spearman"].nunique() >= 2:
            r, p = pearsonr(sub["model_spearman"], sub[metric])
            rho, prho = spearmanr(sub["model_spearman"], sub[metric])
            ax.text(0.0, -0.16,
                    f"n={len(sub)}  ·  Pearson r={r:+.2f} (p={p:.2g})  ·  "
                    f"Spearman ρ={rho:+.2f} (p={prho:.2g})\n{band}  ·  {rb_txt}",
                    transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                    color="#555")

    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.30, 0.014, 0.40])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Noise level (label_noise_std)")

    fig.suptitle(f"{label} (↑ better) vs. model performance — reframed off noise level"
                 f"   ·   red band = random-ranking null (95%)",
                 fontsize=13, fontweight="bold", y=0.955)

    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"shen_style_{metric}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  saved {out_dir.name}/shen_style_{metric}.png / .svg")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/project/aip-bensl/ranaab/IGLOO/results/synthetic"),
                    help="dir containing the {Protein}_{landscape}/noise_*/ folders")
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "attrq_vs_modelperf_noise")
    ap.add_argument("--replot", action="store_true",
                    help="skip the (slow) CSV scan; re-read attrq_vs_modelperf_noise.csv "
                         "and just regenerate figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "attrq_vs_modelperf_noise.csv"

    if args.replot and csv_path.exists():
        print(f"Re-reading {csv_path} (--replot) ...")
        df = pd.read_csv(csv_path)
    else:
        print(f"Scanning {args.root} ...")
        df = collect(args.root)
        if df.empty:
            raise SystemExit("No runs found.")
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}  ({len(df)} rows: "
              f"{df['protein'].nunique()} proteins x {df['landscape'].nunique()} landscapes "
              f"x {df['noise'].nunique()} noise levels)")

    # aggregation-FIRST metrics (position_importance_summary.csv)
    agg_first = [
        ("attr_ndcg", "Global NDCG"),
        ("attr_spearman", "Attribution Spearman ρ (|attr| vs gt importance)"),
        ("attr_auroc", "AUROC (important vs background positions)"),
        ("attr_ap", "Average precision"),
        ("attr_kendall", "Attribution Kendall τ"),
    ]
    # aggregation-LAST metrics (score each variant, then average) + error col
    agg_last = [
        ("pv_ndcg_mean", "Per-variant NDCG", "pv_ndcg_sem"),
        ("pv_spearman_mean", "Per-variant Spearman ρ (|IG| vs gt)", "pv_spearman_sem"),
        ("pv_mut_auroc_mean", "Per-variant mutation AUROC (mutated vs other positions)",
         "pv_mut_auroc_sem"),
    ]

    print("\nGenerating figures (aggregation-first):")
    for metric, label in agg_first:
        plot_metric(df, metric, label, args.out)
        plot_shen_style(df, metric, label, args.out)

    print("\nGenerating figures (aggregation-last / per-variant):")
    for metric, label, errcol in agg_last:
        if metric in df and df[metric].notna().any():
            plot_metric(df, metric, label, args.out)
            plot_shen_style(df, metric, label, args.out, err=errcol)

    print("\nCorrelation of attribution quality with model test performance "
          "(pooled, n per row):")
    for metric, *_ in agg_first + [(m, l) for m, l, _ in agg_last]:
        if metric not in df:
            continue
        sub = df.dropna(subset=["model_spearman", metric])
        if len(sub) < 3 or sub[metric].nunique() < 2:
            continue
        r, p = pearsonr(sub["model_spearman"], sub[metric])
        # within-panel (partialling out the between-condition gap): mean of per-(protein,landscape) r
        wp = []
        for (pr_, ls_), gg in sub.groupby(["protein", "landscape"]):
            if len(gg) >= 3 and gg["model_spearman"].nunique() >= 2:
                wp.append(pearsonr(gg["model_spearman"], gg[metric])[0])
        wp_txt = f", within-panel mean r = {np.mean(wp):+.3f}" if wp else ""
        print(f"  {metric:20s}: pooled Pearson r = {r:+.3f} (p={p:.2g}, n={len(sub)}){wp_txt}")


if __name__ == "__main__":
    main()
