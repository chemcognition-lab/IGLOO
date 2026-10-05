#!/usr/bin/env python3
"""
Illustrative schematic (NOT fit to real generator parameters) of how fitness
declines with more simultaneous mutations under each of the three synthetic
landscape models in src/igloo/synthetic_data.py, assuming all mutations fall
within the same structural/contact region (same SS region, contact=1.0) --
the case where additive, cliff, and epistatic behavior are maximally
distinguishable.

  Additive:   fitness = 1 - d*n                              (straight line)
  Nonlinear:  additive until a damage threshold, then a steep
              multiplier kicks in on the remaining damage      (flat, then cliff)
  Epistatic:  additive MINUS a pairwise term that grows as
              C(n,2) -- zero for n<=1 (no pair yet), then
              accelerating                                     (smooth, curving away)

d (per-mutation damage) and the epistatic/cliff scale constants here are
illustrative round numbers chosen to make the three FUNCTIONAL FORMS visually
distinct -- not fitted to the real blosum_penalty / importance_weight /
epistasis_scale values in synthetic_data.py.
"""
import math

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"additive": "#1f77b4", "epistatic": "#ff7f0e", "nonlinear": "#2ca02c"}
MARKERS = {"additive": "o", "epistatic": "s", "nonlinear": "D"}

n = np.arange(0, 6)
d = 0.15

additive = 1 - d * n

cliff_threshold_n = 3
cliff_drop = 3.0
nonlinear = np.where(
    n <= cliff_threshold_n - 1,
    1 - d * n,
    (1 - d * (cliff_threshold_n - 1)) - cliff_drop * d * (n - (cliff_threshold_n - 1)),
)

pair_k = 0.05
pairs = np.array([math.comb(int(ni), 2) for ni in n])
epistatic = additive - pair_k * pairs

fig, ax = plt.subplots(figsize=(8.5, 6.5))

for name, vals in [("additive", additive), ("epistatic", epistatic), ("nonlinear", nonlinear)]:
    ax.plot(n, vals, f"{MARKERS[name]}-", color=COLORS[name], markersize=9,
            markeredgecolor="black", markeredgewidth=0.8, linewidth=2.2,
            label=name.capitalize(), zorder=3)

ax.axhline(0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)

ax.annotate("identical for a single mutation\n(no pair yet to interact)",
            xy=(1, additive[1]), xytext=(1.5, 0.97),
            fontsize=10.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))

ax.annotate("cliff threshold —\nsudden multiplier kicks in",
            xy=(cliff_threshold_n - 1, nonlinear[cliff_threshold_n - 1]),
            xytext=(3.3, 0.55),
            fontsize=10.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))

ax.annotate("smoothly accelerating —\ndamage grows ∝ C(n,2) pairs,\nonly between contacting positions",
            xy=(4, epistatic[4]), xytext=(0.3, -0.75),
            fontsize=10.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))

ax.set_xticks(n)
ax.set_xlabel("Number of simultaneous mutations\n(all within the same structural/contact region)",
              fontsize=13, color="black")
ax.set_ylabel("Fitness (schematic units)", fontsize=13, color="black")
ax.set_title("Illustrative Shapes of the Three Synthetic Landscapes\n(schematic — not fit to real generator parameters)",
             fontweight="bold", fontsize=14.5, color="black")
ax.tick_params(axis="both", labelsize=11, labelcolor="black", color="black")

legend = ax.legend(loc="upper right", fontsize=12, title="Landscape", title_fontsize=12)
legend.get_frame().set_edgecolor("black")
for text in legend.get_texts():
    text.set_color("black")
legend.get_title().set_color("black")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")

fig.tight_layout()
out = "/tmp/claude-3104396/-project-6007171-ranaab-IGLOO/0388602c-21fd-40cc-a9d5-649fcf1e8c6c/scratchpad/schematics/landscape_shapes_schematic"
fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
print(f"saved {out}.png")
