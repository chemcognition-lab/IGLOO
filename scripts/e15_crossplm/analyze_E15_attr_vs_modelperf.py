#!/usr/bin/env python3
"""
E15: Attribution quality vs model generalization performance.

Uses the data we already have:
  - results/revision/E15_crosspLM/T4_lysozyme/esmc_300m/attributions_*.pkl
        (T4_lysozyme, esmc_300m; 3 landscapes x 5 seeds x 5 methods; 400 test seqs)
  - results/revision/E15_crosspLM/model_performance.csv
        (val_spearman / val_loss per landscape x seed; averaged over reruns)

For every (landscape, seed, method) we average the per-sequence attribution
metrics over the 400 test sequences, then join the model's validation Spearman
for that (landscape, seed).  One point per (landscape, seed); colour = method,
marker = landscape.

NOTE on the noise axis: run_E15_phase1.py computes `fitness_noisy` but never
uses it, so the 7 "noise levels" in metrics_long.csv are the same computation
repeated.  There is therefore no real per-noise model-performance gradient in
the saved data; the meaningful performance spread is *across seeds/landscapes*,
which is what this script plots.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np

# per-sequence attribution rows are occasionally constant -> spearmanr undefined;
# we nan-average over 400 sequences so these are harmless.
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:
    pass
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import ndcg_score, roc_auc_score, average_precision_score

RESULTS_DIR = Path("results/revision/E15_crosspLM")
ATTR_DIR = RESULTS_DIR / "T4_lysozyme" / "esmc_300m"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

METHOD_COLORS = {
    "ig_linear": "#1f77b4",
    "ig_slerp": "#ff7f0e",
    "ig_nearest_neighbor": "#2ca02c",
    "gradient_input": "#d62728",
    "input_difference": "#9467bd",
}
LANDSCAPE_MARKERS = {"additive": "o", "epistatic": "s", "nonlinear": "^"}


def per_sequence_metrics(attr_row: np.ndarray, w: np.ndarray, annotated: np.ndarray) -> dict:
    """Attribution-quality metrics for one sequence's residue attributions."""
    a = attr_row[: len(w)]
    abs_a = np.abs(a)

    sp_signed, _ = spearmanr(a, w)
    sp_abs, _ = spearmanr(abs_a, w)
    kt, _ = kendalltau(abs_a, w)

    ndcg = ndcg_score(w[None, :], abs_a[None, :])

    # gt weights are all positive here -> true sign is +; sign accuracy = frac(attr > 0)
    sign_acc = float(np.mean(np.sign(a) == np.sign(w)))

    if annotated.any() and not annotated.all():
        auroc = roc_auc_score(annotated, abs_a)
        ap_r = average_precision_score(annotated, abs_a)
    else:
        auroc = np.nan
        ap_r = np.nan

    return {
        "spearman_signed": sp_signed,
        "spearman_abs": sp_abs,
        "kendall_abs": kt,
        "ndcg": ndcg,
        "sign_accuracy": sign_acc,
        "auroc": auroc,
        "ap_r": ap_r,
    }


def build_metrics_table() -> pd.DataFrame:
    rows = []
    for pkl in sorted(ATTR_DIR.glob("attributions_T4_lysozyme_*_seed*.pkl")):
        stem = pkl.stem.replace("attributions_T4_lysozyme_", "")
        parts = stem.split("_")
        landscape = parts[0]
        seed = int(parts[-1].replace("seed", ""))

        d = pickle.load(open(pkl, "rb"))
        w = np.asarray(d["ground_truth_w"], dtype=float)
        # gt weight is a sparse importance vector: most positions sit at the
        # background value, a handful are "important". Annotated = above background.
        annotated = w > (w.min() + 1e-9)

        for method, arr in d["all_attributions_residue"].items():
            arr = np.asarray(arr, dtype=float)
            per_seq = [per_sequence_metrics(arr[i], w, annotated) for i in range(arr.shape[0])]
            mean_m = {k: float(np.nanmean([m[k] for m in per_seq])) for k in per_seq[0]}
            rows.append({"landscape": landscape, "seed": seed, "method": method,
                         "n_seqs": arr.shape[0], **mean_m})
    return pd.DataFrame(rows)


def load_model_perf() -> pd.DataFrame:
    mp = pd.read_csv(RESULTS_DIR / "model_performance.csv")
    return (mp.groupby(["landscape", "seed"])
              .agg(val_spearman=("val_spearman", "mean"),
                   val_spearman_std=("val_spearman", "std"),
                   val_loss=("val_loss", "mean"),
                   n_reruns=("val_spearman", "size"))
              .reset_index())


def scatter_metric(df: pd.DataFrame, metric: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    method_handles = []
    for method, g in df.groupby("method"):
        color = METHOD_COLORS.get(method, "#7f7f7f")
        for ls, gg in g.groupby("landscape"):
            ax.scatter(gg["val_spearman"], gg[metric], c=color,
                       marker=LANDSCAPE_MARKERS.get(ls, "o"), s=55,
                       edgecolors="white", linewidths=0.6, alpha=0.9)
        # trend across the (landscape, seed) points for this method
        sub = g.dropna(subset=["val_spearman", metric])
        lbl = method
        if len(sub) >= 3 and sub["val_spearman"].nunique() >= 2 and sub[metric].nunique() >= 2:
            r, p = pearsonr(sub["val_spearman"], sub[metric])
            z = np.polyfit(sub["val_spearman"], sub[metric], 1)
            xs = np.linspace(sub["val_spearman"].min(), sub["val_spearman"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=color, lw=1.8, alpha=0.75)
            lbl = f"{method}   r={r:+.2f} (p={p:.2f})"
        method_handles.append(plt.Line2D([], [], marker="o", ls="-", color=color, label=lbl))

    ls_handles = [plt.Line2D([], [], marker=m, ls="", color="#555", label=ls)
                  for ls, m in LANDSCAPE_MARKERS.items()]

    ax.set_xlabel("Model validation Spearman ρ  (per landscape × seed, mean of reruns)")
    ax.set_ylabel(label)
    ax.set_title(f"T4-lysozyme / ESM-C 300M — {label}\nvs model generalization performance",
                 fontsize=11)
    ax.axhline(0, color="#bbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbb", lw=0.8, zorder=0)

    fig.subplots_adjust(right=0.66)
    leg1 = fig.legend(handles=method_handles, fontsize=8.5, loc="upper left",
                      bbox_to_anchor=(0.68, 0.90),
                      title="method  (trend of metric vs model perf)")
    leg1._legend_box.align = "left"
    fig.legend(handles=ls_handles, fontsize=8.5, loc="upper left",
               bbox_to_anchor=(0.68, 0.50), title="landscape")

    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"attrq_vs_modelperf_{metric}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  saved figures/attrq_vs_modelperf_{metric}.png / .svg")


def main() -> None:
    print("Computing attribution metrics from 15 attribution pkls ...")
    metrics = build_metrics_table()
    perf = load_model_perf()
    df = metrics.merge(perf, on=["landscape", "seed"], how="left")

    out_csv = RESULTS_DIR / "attrq_vs_modelperf.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} rows: "
          f"{df['landscape'].nunique()} landscapes x {df['seed'].nunique()} seeds "
          f"x {df['method'].nunique()} methods)")

    print("\nGenerating scatter plots (attribution quality vs model val Spearman):")
    for metric, label in [
        ("spearman_abs", "Attribution Spearman ρ (|attr| vs gt importance)"),
        ("ndcg", "NDCG"),
        ("auroc", "AUROC (important vs background positions)"),
        ("ap_r", "Average precision"),
        ("sign_accuracy", "Sign accuracy"),
        ("kendall_abs", "Attribution Kendall τ"),
    ]:
        scatter_metric(df, metric, label)

    print("\nCorrelation of attribution quality with model performance "
          "(pooled over methods, 15 points each):")
    for metric in ["spearman_abs", "ndcg", "auroc", "ap_r", "sign_accuracy"]:
        sub = df.dropna(subset=["val_spearman", metric])
        if len(sub) >= 2 and sub[metric].nunique() >= 2:
            r, p = pearsonr(sub["val_spearman"], sub[metric])
            print(f"  {metric:16s}: r = {r:+.3f}  (p = {p:.3f}, n = {len(sub)})")
        else:
            print(f"  {metric:16s}: n/a (insufficient variation, n = {len(sub)})")


if __name__ == "__main__":
    main()
