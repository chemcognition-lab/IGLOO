#!/usr/bin/env python3
"""
k-predictor test, for ISM vs IG attributions.

For each (protein, landscape, noise) with >=2 seeds, compute the mean pairwise
Spearman across seeds of:
  - test-set predictions            (from IG results CSV, rows tagged [test])
  - IG position-importance vector
  - ISM position vector

If seeds agree much more on ISM rankings than on IG rankings, ISM is the less
under-determined attribution.

Reads results/synthetic/seedsweep/ (IG + predictions) and
results/synthetic/ism_attribution/ism_position_*__seed_*.csv (ISM).

Usage: python scripts/ism_seed_agreement.py [--root ...] [--out ...]
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


def _mps(vectors):
    rs = [spearmanr(a, b)[0] for a, b in itertools.combinations(vectors, 2)]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else np.nan


def main():
    project_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/project/aip-bensl/ranaab/IGLOO/results/synthetic"))
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "ism_seed_agreement")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ism_dir = args.root / "ism_attribution"

    rows = []
    for p in PROTEINS:
        for l in LANDSCAPES:
            base = args.root / "seedsweep" / f"{p}_{l}"
            if not base.is_dir():
                continue
            for nd in sorted(base.glob("noise_*")):
                noise = float(nd.name.split("_", 1)[1])
                seeds = sorted(nd.glob("seed_*/linear"))
                preds, igvecs, common = [], [], None
                for sd in seeds:
                    f = list(sd.glob("*_integrated_gradients_results.csv"))
                    s = list(sd.glob("*_position_importance_summary.csv"))
                    if not f or not s:
                        continue
                    df = pd.read_csv(f[0], usecols=["sequence_name", "predicted_activity"])
                    t = df[df["sequence_name"].str.contains(r"\[test\]", na=False)]
                    pr = t.groupby("sequence_name")["predicted_activity"].first()
                    preds.append(pr)
                    igvecs.append(pd.read_csv(s[0]).sort_values("position")["importance_score"].to_numpy())
                    common = set(pr.index) if common is None else common & set(pr.index)
                # ISM seed vectors for this cell
                ismvecs = []
                for g in [f"ism_position_{p}_{l}__noise_{noise:g}__seed_*.csv",
                          f"ism_position_{p}_{l}__noise_{noise:.1f}__seed_*.csv"]:
                    for f in sorted(ism_dir.glob(g)):
                        ismvecs.append(pd.read_csv(f).sort_values("position")["ism_mean_abs"].to_numpy())
                    if ismvecs:
                        break

                rec = {"protein": p, "landscape": l, "noise": noise,
                       "n_ig_seeds": len(igvecs), "n_ism_seeds": len(ismvecs)}
                if len(preds) >= 2 and common:
                    common = sorted(common)
                    rec["prediction_agreement"] = _mps([x.reindex(common).to_numpy() for x in preds])
                if len(igvecs) >= 2:
                    rec["ig_attr_agreement"] = _mps(igvecs)
                if len(ismvecs) >= 2:
                    rec["ism_attr_agreement"] = _mps(ismvecs)
                rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "ism_seed_agreement.csv", index=False)
    keep = df.dropna(subset=["ig_attr_agreement", "ism_attr_agreement"], how="all")
    print(keep.to_string(index=False))

    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].dropna(
                  subset=["ig_attr_agreement", "ism_attr_agreement"], how="all").empty]
    if not combos:
        print("\nnot enough seed data yet"); return
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow),
                             squeeze=False, layout="constrained")
    for ax, (p, l) in zip(axes.flat, combos):
        g = df[(df.protein == p) & (df.landscape == l)].sort_values("noise")
        for col, c, lab in [("prediction_agreement", "#2ca02c", "predictions"),
                            ("ig_attr_agreement", "#d62728", "IG rankings"),
                            ("ism_attr_agreement", "#1f77b4", "ISM rankings")]:
            if col in g and g[col].notna().any():
                ax.plot(g["noise"], g[col], "o-", color=c, label=lab)
        ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
        ax.set_xlabel("label noise (std)")
        ax.set_ylabel("mean pairwise Spearman ρ across 5 seeds")
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    fig.suptitle("Do the 5 seeds agree more on ISM rankings than on IG rankings?",
                 fontsize=13, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"ism_seed_agreement.{ext}", dpi=150)
    plt.close(fig)
    print(f"\nsaved {args.out}/ism_seed_agreement.png")


if __name__ == "__main__":
    main()
