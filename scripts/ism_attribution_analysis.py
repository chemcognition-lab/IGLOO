#!/usr/bin/env python3
"""
Does ISM attribution track model performance where IG does not?

Reads results/synthetic/ism_attribution/ism_summary_*.csv and overlays the ISM
attribution quality (Spearman |ISM| vs w, NDCG) against the IG version from
results/synthetic/attrq_vs_modelperf_noise/attrq_vs_modelperf_noise.csv, both
plotted vs model test Spearman and coloured by noise.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent / "results" / "synthetic"
ISM_DIR = ROOT / "ism_attribution"
OUT = ROOT / "ism_attribution"
OUT.mkdir(parents=True, exist_ok=True)

sfiles = sorted(ISM_DIR.glob("ism_summary_*.csv"))
if not sfiles:
    raise SystemExit("no ism_summary_*.csv yet — run the ism_* jobs first")
ism = pd.concat([pd.read_csv(f) for f in sfiles], ignore_index=True)
ism.to_csv(OUT / "ism_summary_all.csv", index=False)
print(f"{len(ism)} ISM runs")

ig_csv = ROOT / "attrq_vs_modelperf_noise" / "attrq_vs_modelperf_noise.csv"
ig = pd.read_csv(ig_csv) if ig_csv.exists() else None

proteins = sorted(ism["protein"].unique())
landscapes = ["additive", "epistatic", "nonlinear"]
combos = [(p, l) for p in proteins for l in landscapes
          if not ism[(ism.protein == p) & (ism.landscape == l)].empty]
ncol = 3
nrow = int(np.ceil(len(combos) / ncol))
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(0, 1)

for ismcol, igcol, lab in [
    ("ism_spearman_vs_w", "attr_spearman", "Spearman(|attr|, ground-truth w)"),
    ("ism_ndcg", "attr_ndcg", "NDCG"),
]:
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.4 * nrow),
                             squeeze=False, layout="constrained")
    for ax, (p, l) in zip(axes.flat, combos):
        gi = ism[(ism.protein == p) & (ism.landscape == l)].sort_values("noise")
        ax.plot(gi["model_test_spearman"], gi[ismcol], "-", color="#444", lw=1, zorder=1)
        ax.scatter(gi["model_test_spearman"], gi[ismcol], c=gi["noise"], cmap=cmap,
                   norm=norm, s=90, edgecolors="black", linewidths=0.5, zorder=3,
                   label="ISM")
        if ig is not None:
            gg = ig[(ig.protein == p) & (ig.landscape == l)].sort_values("noise")
            if not gg.empty:
                ax.plot(gg["model_spearman"], gg[igcol], "--", color="#d62728",
                        lw=1.2, marker="x", ms=7, label="IG (signed)", zorder=2)
        r = np.nan
        sub = gi.dropna(subset=["model_test_spearman", ismcol])
        if len(sub) >= 3:
            r = pearsonr(sub["model_test_spearman"], sub[ismcol])[0]
        ax.set_title(f"{p} — {l}   (ISM r vs model perf = {r:+.2f})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Model test performance (Spearman ρ)")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, label="label noise (std)")
    fig.suptitle(f"ISM (model-dependent) vs IG attribution — {lab} vs model performance",
                 fontsize=13, fontweight="bold")
    tag = ismcol.replace("ism_", "")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"ism_vs_ig_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"saved ism_attribution/ism_vs_ig_{tag}.png")

# headline numbers
print("\nISM Spearman(|ISM|, w) vs model test performance (pooled):")
for (p, l), g in ism.groupby(["protein", "landscape"]):
    s = g.dropna(subset=["model_test_spearman", "ism_spearman_vs_w"])
    if len(s) >= 3:
        r, pv = pearsonr(s["model_test_spearman"], s["ism_spearman_vs_w"])
        print(f"  {p:11s} {l:10s}: r = {r:+.3f} (p={pv:.2g}, n={len(s)})  "
              f"ISM ρ range [{g.ism_spearman_vs_w.min():.2f}, {g.ism_spearman_vs_w.max():.2f}]")
