#!/usr/bin/env python3
"""
Path-2: train k predictors (seeds) per condition, compare how much they agree on
TEST-SET PREDICTIONS vs. how much they agree on ATTRIBUTION RANKINGS.

If the k models agree on predictions but disagree on attributions (or vice
versa) that tells us whether attribution reflects the learned predictor or
something model-independent (pLM geometry).

Reads the multi-seed sweep:
    <root>/seedsweep/{Protein}_{landscape}/noise_{N}/seed_{S}/linear/
        *_integrated_gradients_results.csv   -> per-variant predicted_activity
                                                (test rows tagged '[test]') and
                                                per-position integrated_gradient
        *_position_importance_summary.csv    -> aggregate |IG| per position

Per (protein, landscape, noise):
  prediction_agreement = mean over seed-pairs of Spearman(pred_a, pred_b) on the
                         common set of test variants
  attribution_agreement = mean over seed-pairs of Spearman(posvec_a, posvec_b)

Also joins per-seed model test Spearman (parsed from run logs) so the x-axis can
be model performance.

Usage:
    python scripts/sanity_checks/k_predictor_agreement.py \
        [--root /project/aip-bensl/ranaab/IGLOO/results/synthetic] \
        [--out  results/synthetic/k_predictor_agreement]
"""

import argparse
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:
    pass

PROTEINS = ["GB1", "T4Lysozyme"]
LANDSCAPES = ["additive", "epistatic", "nonlinear"]
_SP_RE = re.compile(r"Spearman\s*[ρr]\s*=\s*(-?\d+\.\d+)")


def _model_test_spearman(seed_dir: Path) -> float:
    logs = sorted(seed_dir.glob("*_run_log_*.txt"))
    if not logs:
        return np.nan
    m = _SP_RE.findall(logs[-1].read_text(errors="ignore"))
    return float(m[-1]) if m else np.nan


def _seed_vectors(seed_dir: Path):
    """Return (test_pred Series indexed by variant, position-importance vector)."""
    ig = list(seed_dir.glob("*_integrated_gradients_results.csv"))
    summ = list(seed_dir.glob("*_position_importance_summary.csv"))
    if not ig or not summ:
        return None, None
    df = pd.read_csv(ig[0], usecols=["sequence_name", "predicted_activity"])
    test = df[df["sequence_name"].str.contains(r"\[test\]", na=False)]
    # one row per variant
    pred = test.groupby("sequence_name")["predicted_activity"].first()
    pos = (pd.read_csv(summ[0]).sort_values("position")["importance_score"].to_numpy())
    return pred, pos


def _mean_pairwise_spearman(vectors: list[np.ndarray]) -> tuple[float, int]:
    rs = []
    for a, b in itertools.combinations(vectors, 2):
        r, _ = spearmanr(a, b)
        if np.isfinite(r):
            rs.append(r)
    return (float(np.mean(rs)) if rs else np.nan), len(rs)


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for protein in PROTEINS:
        for landscape in LANDSCAPES:
            base = root / "seedsweep" / f"{protein}_{landscape}"
            if not base.is_dir():
                continue
            for noise_dir in sorted(base.glob("noise_*")):
                noise = float(noise_dir.name.split("_", 1)[1])
                seed_dirs = sorted(noise_dir.glob("seed_*/linear"))
                preds, posvecs, perfs = [], [], []
                common = None
                for sd in seed_dirs:
                    pred, pos = _seed_vectors(sd)
                    if pred is None:
                        continue
                    preds.append(pred)
                    posvecs.append(pos)
                    perfs.append(_model_test_spearman(sd))
                    common = set(pred.index) if common is None else (common & set(pred.index))
                if len(preds) < 2 or not common:
                    continue
                common = sorted(common)
                pred_mat = [p.reindex(common).to_numpy() for p in preds]
                pred_agree, npair = _mean_pairwise_spearman(pred_mat)
                attr_agree, _ = _mean_pairwise_spearman(posvecs)
                rows.append({
                    "protein": protein, "landscape": landscape, "noise": noise,
                    "n_seeds": len(preds), "n_pairs": npair,
                    "n_common_test_variants": len(common),
                    "model_spearman_mean": float(np.nanmean(perfs)),
                    "model_spearman_std": float(np.nanstd(perfs)),
                    "prediction_agreement": pred_agree,
                    "attribution_agreement": attr_agree,
                })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_dir: Path) -> None:
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    if not combos:
        print("  nothing to plot yet")
        return
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    for xaxis, xlabel, xcol in [
        ("noise", "Label noise (std)", "noise"),
        ("modelperf", "Mean model test Spearman ρ", "model_spearman_mean"),
    ]:
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                                 squeeze=False, layout="constrained")
        for ax, (p, l) in zip(axes.flat, combos):
            g = df[(df.protein == p) & (df.landscape == l)].sort_values(xcol)
            ax.plot(g[xcol], g["prediction_agreement"], "o-", color="#1f77b4",
                    label="prediction agreement (test set)")
            ax.plot(g[xcol], g["attribution_agreement"], "s-", color="#d62728",
                    label="attribution-ranking agreement")
            ax.set_title(f"{p} — {l}", fontweight="bold", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("mean pairwise Spearman ρ across seeds")
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(0, color="#ccc", lw=0.8)
            ax.legend(fontsize=8)
        for ax in axes.flat[len(combos):]:
            ax.set_visible(False)
        fig.suptitle("k-predictor agreement: do the seeds agree on predictions "
                     "vs on attributions?", fontsize=13, fontweight="bold")
        for ext in ("png", "svg"):
            fig.savefig(out_dir / f"k_predictor_agreement_by_{xaxis}.{ext}", dpi=150)
        plt.close(fig)
        print(f"  saved {out_dir.name}/k_predictor_agreement_by_{xaxis}.png / .svg")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/project/aip-bensl/ranaab/IGLOO/results/synthetic"))
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "k_predictor_agreement")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.root}/seedsweep ...")
    df = collect(args.root)
    if df.empty:
        print("No completed seed-sweep conditions yet (need >=2 seeds per cell). "
              "Re-run this once the sweep progresses.")
        return
    csv_path = args.out / "k_predictor_agreement.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} conditions)")
    print(df[["protein", "landscape", "noise", "n_seeds",
              "prediction_agreement", "attribution_agreement"]].to_string(index=False))
    plot(df, args.out)


if __name__ == "__main__":
    main()
