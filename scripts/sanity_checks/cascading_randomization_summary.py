#!/usr/bin/env python3
"""Aggregate the per-condition cascading_*.csv files into one figure."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[2] / "results" / "synthetic" / "cascading_randomization"
files = sorted(D.glob("cascading_*.csv"))
if not files:
    raise SystemExit("no cascading_*.csv yet")

frames = []
for f in files:
    df = pd.read_csv(f)
    m = re.match(r"cascading_([A-Za-z0-9]+)_([a-z]+)__noise_([0-9.]+)", f.stem)
    if not m:
        continue
    df["protein"], df["landscape"], df["noise"] = m.group(1), m.group(2), float(m.group(3))
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
df.to_csv(D / "cascading_summary.csv", index=False)

proteins = sorted(df["protein"].unique())
landscapes = ["additive", "epistatic", "nonlinear"]
combos = [(p, l) for p in proteins for l in landscapes
          if not df[(df.protein == p) & (df.landscape == l)].empty]
ncol = 3
nrow = int(np.ceil(len(combos) / ncol))
cmap = plt.get_cmap("viridis")
noises = sorted(df["noise"].unique())

for yc, ylab in [("spearman_vs_original", "Spearman(attr, ORIGINAL attr)"),
                 ("spearman_vs_gt", "Spearman(attr, ground-truth w)")]:
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.4 * nrow),
                             squeeze=False, layout="constrained")
    for ax, (p, l) in zip(axes.flat, combos):
        g = df[(df.protein == p) & (df.landscape == l)]
        for nz in noises:
            gg = g[g.noise == nz].sort_values("level")
            if gg.empty:
                continue
            ax.plot(gg["level"], gg[yc], "o-", color=cmap(nz),
                    label=f"noise {nz:g}")
        ax.set_title(f"{p} — {l}", fontweight="bold", fontsize=10)
        ax.set_xticks(sorted(df["level"].unique()))
        labs = (df[df.protein == p].sort_values("level")
                .drop_duplicates("level")["label"].tolist())
        ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_ylim(-0.35, 1.05)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    fig.suptitle("Cascading model-parameter randomization (output→input)\n"
                 "flat curve ⇒ attribution not driven by the learned predictor",
                 fontsize=13, fontweight="bold")
    tag = yc.replace("spearman_vs_", "")
    for ext in ("png", "svg"):
        fig.savefig(D / f"cascading_summary_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"saved cascading_summary_{tag}.png / .svg")
