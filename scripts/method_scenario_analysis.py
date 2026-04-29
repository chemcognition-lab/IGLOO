#!/usr/bin/env python3
"""
Method-Scenario Interaction Analysis for IGLOO IG Attribution Quality
======================================================================
Systematically answers: "Which interpolation method works best, and does
the answer depend on the protein / fitness landscape (scenario)?"

Pipeline
--------
1.  Discover results and compute metrics via ig_metrics.compute_metrics_for_dir
2.  Build normalised composite score matrix  (methods × scenarios × noise_levels)
3.  Summarise noise robustness:  AUC over noise → (methods × scenarios) + exponential decay fit
4.  Friedman test  – do methods differ significantly at all?
5.  Nemenyi post-hoc  – which pairs are significantly different?
6.  Critical Difference (CD) diagram  (Demšar 2006)
7.  Kendall's W  – are scenario rankings concordant?
8.  If W < threshold → decision tree on scenario features
9.  Save a summary CSV + all figures

Usage
-----
    # auto-discover everything
    python scripts/method_scenario_analysis.py

    # with filters
    python scripts/method_scenario_analysis.py \\
        --proteins GB1 T4Lysozyme \\
        --fitness-models additive epistatic nonlinear \\
        --results-dir results/synthetic \\
        --output-dir  results/synthetic/method_analysis

    # multi-seed layout
    python scripts/method_scenario_analysis.py --multi-seed

Flags
-----
    --results-dir PATH     Root of synthetic results   (default: results/synthetic)
    --output-dir  PATH     Output directory for figures (default: results/synthetic/method_analysis)
    --proteins    [...]    Restrict proteins
    --fitness-models [...] Restrict fitness models
    --metrics     [...]    Metrics to include in composite score
                           (default: spearman auroc auprc prec_k pv_mean sign_acc cat_sep)
    --concordance-threshold FLOAT
                           Kendall's W below which we fit a decision tree (default: 0.7)
    --multi-seed           Use discover_result_dirs_multiseed layout
    --min-output-diff FLOAT
    --min-count INT
    --dpi INT
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import friedmanchisquare, rankdata

# ──────────────────────────────────────────────────────────────────────────────
# Project layout
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "results" / "synthetic"
DEFAULT_OUTPUT_DIR  = _PROJECT_ROOT / "results" / "synthetic" / "method_analysis"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.ig_metrics import (
    NEAR_ZERO_THRESHOLD,
    _KNOWN_PROTEINS,
    _PROTEIN_DEFS,
    discover_result_dirs,
    discover_result_dirs_multiseed,
    compute_metrics_for_dir,
    _agg_metrics,
    _load_gt_json,
)
from src.igloo.plot import INTERP_COLORS, INTERP_LABELS, INTERP_MARKERS

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Metrics included in the composite score.
# All should be in [0,1] or have a natural direction (higher = better).
DEFAULT_METRICS: list[str] = [
    "spearman",      # Global position-level Spearman ρ (abs IG track)
    "auroc",         # AUROC for annotated vs background
    "auprc",         # AUPRC
    "prec_k",        # Precision@K
    "pv_mean",       # Mean per-variant Spearman ρ
    "sign_acc",      # Sign accuracy
    "cat_sep",       # Category separation ratio
]

# Scenario features for the decision tree
SCENARIO_FEATURES: dict[str, str] = {
    "protein":             "categorical",
    "fitness_model":       "categorical",
    "sequence_length":     "continuous",
    "annotation_density":  "continuous",   # n_annotated / sequence_length
    "n_annotated":         "continuous",
    "max_category_weight": "continuous",
}

FIG_BG = "#F7F8FA"
matplotlib.rcParams.update({
    "figure.facecolor":  FIG_BG,
    "axes.facecolor":    FIG_BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         9,
})


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load / compute metrics
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics(
    results_root: Path,
    proteins: list[str] | None,
    fitness_models: list[str] | None,
    min_output_diff: float,
    min_count: int,
    multi_seed: bool,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
        protein, fitness_model, noise, interp, <metric_cols>...
    Each row is one (protein, fitness_model, noise, interp) leaf.
    For multi-seed, metrics are averaged across seeds.
    """
    rows: list[dict] = []

    if multi_seed:
        entries = discover_result_dirs_multiseed(
            results_root, proteins=proteins, fitness_models=fitness_models
        )
        if not entries:
            return pd.DataFrame()
        # Buffer: (protein, fm, noise, interp) → list of metric dicts
        buf: dict[tuple, list[dict]] = {}
        for protein, fm, noise, _seed, interp, path in entries:
            key = (protein, fm, noise, interp)
            m = compute_metrics_for_dir(
                path, protein,
                min_output_diff=min_output_diff,
                min_count=min_count,
                fitness_model=fm,
            )
            if m is not None:
                buf.setdefault(key, []).append(m)
        for (protein, fm, noise, interp), mlist in buf.items():
            mean_m, _ = _agg_metrics(mlist)
            row = {"protein": protein, "fitness_model": fm,
                   "noise": noise, "interp": interp}
            row.update(mean_m)
            rows.append(row)
    else:
        entries = discover_result_dirs(
            results_root, proteins=proteins, fitness_models=fitness_models
        )
        for protein, fm, noise, interp, path in entries:
            print(f"  Loading {protein}/{fm}/noise={noise:.1f}/{interp}...", end=" ", flush=True)
            m = compute_metrics_for_dir(
                path, protein,
                min_output_diff=min_output_diff,
                min_count=min_count,
                fitness_model=fm,
            )
            if m is None:
                print("SKIP")
                continue
            row = {"protein": protein, "fitness_model": fm,
                   "noise": noise, "interp": interp}
            row.update(m)
            print(f"sp={m['spearman']:+.3f}")
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Drop list/dict columns that come from metrics
    scalar_cols = [c for c in df.columns if df[c].dtype != object or c in
                   ("protein", "fitness_model", "interp")]
    return df[scalar_cols]


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Normalised composite score
# ──────────────────────────────────────────────────────────────────────────────

def compute_composite_scores(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    For each (scenario, noise_level), min-max normalise each metric across
    methods, then average normalised values to get a composite score.

    Returns df with an added 'composite' column (and one 'norm_{m}' column
    per metric).
    """
    scenario_key = ["protein", "fitness_model", "noise"]
    available    = [m for m in metrics if m in df.columns]
    if not available:
        raise ValueError(f"None of the requested metrics found in DataFrame. "
                         f"Available: {list(df.columns)}")
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        warnings.warn(f"Metrics not found and skipped: {missing}")

    df = df.copy()
    norm_dfs: list[pd.DataFrame] = []
    for _, grp in df.groupby(scenario_key):
        grp = grp.copy()
        for m in available:
            lo = grp[m].min()
            hi = grp[m].max()
            denom = hi - lo
            if denom < 1e-12:
                grp[f"norm_{m}"] = 0.5  # all methods identical on this metric
            else:
                grp[f"norm_{m}"] = (grp[m] - lo) / denom
        grp["composite"] = grp[[f"norm_{m}" for m in available]].mean(axis=1)
        norm_dfs.append(grp)

    return pd.concat(norm_dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Noise robustness: AUC over noise + exponential decay fit
# ──────────────────────────────────────────────────────────────────────────────

def _exp_decay(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """score(n) ≈ a * exp(-b * n) + c"""
    return a * np.exp(-b * n) + c


def noise_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (protein, fitness_model, interp):
      - AUC over noise (trapezoidal, normalised by noise range)
      - Peak performance (score at lowest noise level)
      - Floor performance (score at highest noise level)
      - Decay rate  b from exp decay fit (NaN if fit fails)

    Returns a DataFrame indexed by (protein, fitness_model, interp).
    """
    rows: list[dict] = []
    group_cols = ["protein", "fitness_model", "interp"]
    for (protein, fm, interp), grp in df.groupby(group_cols):
        grp  = grp.sort_values("noise")
        ns   = grp["noise"].values.astype(float)
        sc   = grp["composite"].values.astype(float)

        # AUC (trapz), normalised
        noise_range = ns[-1] - ns[0] if len(ns) > 1 else 1.0
        auc = float(np.trapz(sc, ns) / max(noise_range, 1e-9)) if len(ns) > 1 else float(sc[0])

        peak  = float(sc[0])
        floor = float(sc[-1])

        # Exponential decay fit
        decay_rate = float("nan")
        try:
            if len(ns) >= 3:
                p0 = (peak - floor, 1.0, floor)
                popt, _ = curve_fit(
                    _exp_decay, ns, sc, p0=p0,
                    bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                    maxfev=2000,
                )
                decay_rate = float(popt[1])
        except Exception:
            pass

        rows.append({
            "protein": protein, "fitness_model": fm, "interp": interp,
            "auc_noise": auc,
            "peak_perf": peak,
            "floor_perf": floor,
            "decay_rate": decay_rate,
            "n_noise_levels": len(ns),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Friedman test
# ──────────────────────────────────────────────────────────────────────────────

def run_friedman(auc_df: pd.DataFrame) -> dict:
    """
    Friedman test across interpolation methods.

    auc_df has columns (protein, fitness_model, interp, auc_noise).
    Scenarios = (protein, fitness_model) combinations.
    Returns dict with stat, pvalue, and a rank_df.
    """
    pivot = auc_df.pivot_table(
        index=["protein", "fitness_model"],
        columns="interp",
        values="auc_noise",
    )
    pivot = pivot.dropna()
    if pivot.empty or pivot.shape[1] < 2:
        return {"stat": float("nan"), "pvalue": float("nan"), "rank_df": None, "pivot": pivot}

    data = [pivot[col].values for col in pivot.columns]
    stat, pvalue = friedmanchisquare(*data)

    # Compute average ranks (rank 1 = best)
    rank_mat = pivot.rank(axis=1, ascending=False)  # higher AUC → lower rank
    avg_ranks = rank_mat.mean(axis=0).sort_values()

    rank_df = pd.DataFrame({
        "method":   avg_ranks.index,
        "avg_rank": avg_ranks.values,
    })

    return {
        "stat":     float(stat),
        "pvalue":   float(pvalue),
        "rank_df":  rank_df,
        "pivot":    pivot,
        "rank_mat": rank_mat,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Nemenyi post-hoc (self-contained)
# ──────────────────────────────────────────────────────────────────────────────

def nemenyi_posthoc(rank_mat: pd.DataFrame) -> pd.DataFrame:
    """
    Nemenyi post-hoc test for Friedman-ranked data.

    Computes pairwise p-values using the normal approximation:
        z = |R_i - R_j| / sqrt( k(k+1) / (6*N) )
    where k = number of methods, N = number of scenarios (blocks).

    Returns a square DataFrame of p-values.
    """
    k = rank_mat.shape[1]
    N = rank_mat.shape[0]
    avg_ranks = rank_mat.mean(axis=0)
    methods   = list(avg_ranks.index)
    se = np.sqrt(k * (k + 1) / (6 * N))

    pmat = pd.DataFrame(np.ones((k, k)), index=methods, columns=methods)
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(avg_ranks.iloc[i] - avg_ranks.iloc[j]) / se
            # Two-sided normal p-value
            p = float(2 * (1 - stats.norm.cdf(z)))
            pmat.iloc[i, j] = p
            pmat.iloc[j, i] = p
    return pmat


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — CD diagram
# ──────────────────────────────────────────────────────────────────────────────

def plot_cd_diagram(
    rank_df: pd.DataFrame,
    pmat: pd.DataFrame,
    n_scenarios: int,
    alpha: float = 0.05,
    output_path: Path | None = None,
    dpi: int = 150,
) -> None:
    """
    Demšar (2006) Critical Difference diagram.

    Methods are plotted by average rank on a horizontal axis (lower = better,
    left). Horizontal bars connect groups of methods that are NOT significantly
    different from each other.
    """
    methods = rank_df["method"].tolist()
    avg_ranks = rank_df["avg_rank"].tolist()

    k = len(methods)
    n_methods = len(methods)

    # CD at given alpha (Nemenyi): CD = q_alpha * sqrt(k(k+1)/6N)
    # Critical q values (two-tailed) from Demšar Table 5
    _Q = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
          7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = _Q.get(k, 2.728)
    cd = q * np.sqrt(k * (k + 1) / (6 * n_scenarios))

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * k), 3), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.spines[:].set_visible(False)
    ax.set_yticks([])

    rank_lo = 1.0
    rank_hi = float(k)
    ax.set_xlim(rank_lo - 0.5, rank_hi + 0.5)
    ax.set_xlabel("Average rank  (lower = better)", fontsize=9)

    # Draw axis line
    ax.axhline(0.85, color="black", linewidth=1.5,
               xmin=(rank_lo - 0.5 - rank_lo + 0.5) / (rank_hi - rank_lo + 1),
               xmax=1.0)

    y_base = 0.8
    y_label = 0.55

    label_map = INTERP_LABELS
    color_map = INTERP_COLORS

    for i, (m, r) in enumerate(zip(methods, avg_ranks)):
        col = color_map.get(m, "#607D8B")
        ax.plot(r, y_base, marker=INTERP_MARKERS.get(m, "o"),
                color=col, markersize=10, zorder=3)
        lbl = label_map.get(m, m)
        ax.text(r, y_label, lbl, ha="center", va="top", fontsize=8, color=col, fontweight="bold")

    # Draw CD bar (reference)
    # Placed above the diagram as a legend
    bar_y = 1.05
    cd_x  = rank_lo
    ax.annotate(
        "", xy=(cd_x + cd, bar_y), xytext=(cd_x, bar_y),
        arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.5),
    )
    ax.text(cd_x + cd / 2, bar_y + 0.04, f"CD={cd:.2f}", ha="center", fontsize=8, color="dimgray")

    # Draw clique bars (methods not significantly different)
    cliques = _find_cliques(methods, avg_ranks, pmat, alpha)
    bar_ys = np.linspace(0.87, 0.94, max(len(cliques), 1))
    for clique, by in zip(cliques, bar_ys):
        ranks_in = [avg_ranks[methods.index(m)] for m in clique]
        ax.plot([min(ranks_in), max(ranks_in)], [by, by],
                color="black", linewidth=3, solid_capstyle="round", alpha=0.5)

    ax.set_ylim(0.3, 1.2)
    ax.set_title(
        f"Critical Difference diagram  "
        f"(Nemenyi α={alpha}, N={n_scenarios} scenarios)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path.name}")
    plt.close(fig)


def _find_cliques(
    methods: list[str],
    avg_ranks: list[float],
    pmat: pd.DataFrame,
    alpha: float,
) -> list[list[str]]:
    """
    Return groups (cliques) of methods that are mutually NOT significantly
    different (p > alpha for all pairs within the group).
    Uses a greedy spanning approach: sort by rank, then extend cliques.
    """
    order = np.argsort(avg_ranks)
    sorted_methods = [methods[i] for i in order]
    cliques: list[list[str]] = []
    used = [False] * len(sorted_methods)
    for i, m_i in enumerate(sorted_methods):
        if used[i]:
            continue
        clique = [m_i]
        for j in range(i + 1, len(sorted_methods)):
            m_j = sorted_methods[j]
            if all(pmat.loc[m_c, m_j] > alpha for m_c in clique):
                clique.append(m_j)
                used[j] = True
        if len(clique) > 1:
            cliques.append(clique)
    return cliques


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — Kendall's W
# ──────────────────────────────────────────────────────────────────────────────

def kendalls_w(rank_mat: pd.DataFrame) -> float:
    """
    Kendall's W (concordance coefficient).

    rank_mat: rows = scenarios, columns = methods, values = ranks (1 = best).
    W = 1 → perfect concordance.  W = 0 → no concordance.
    """
    N, k = rank_mat.shape
    R = rank_mat.values.astype(float)
    R_mean = R.mean(axis=1, keepdims=True)
    SS = float(np.sum((R.sum(axis=0) - N * (k + 1) / 2) ** 2))
    W = 12 * SS / (N ** 2 * (k ** 3 - k))
    return float(np.clip(W, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Step 8 — Scenario features + decision tree
# ──────────────────────────────────────────────────────────────────────────────

def build_scenario_features(known_proteins: list[str]) -> pd.DataFrame:
    """
    Build a feature table for all protein × fitness_model combinations.
    """
    rows: list[dict] = []
    for protein in known_proteins:
        try:
            gt = _load_gt_json(protein)
        except FileNotFoundError:
            continue
        L = int(gt["sequence_length"])
        ann = gt["annotations"]         # int → category
        cat_w = gt.get("category_weights", {})
        n_ann = len(ann)
        max_w = float(max(cat_w.values())) if cat_w else float("nan")
        ann_density = n_ann / L if L > 0 else float("nan")

        prot_def = _PROTEIN_DEFS.get(protein)
        has_cliff = (
            prot_def is not None
            and any(c in ("active_site", "metal_binding")
                    for c in getattr(prot_def, "annotations", {}).values())
        )

        for fm in ("additive", "epistatic", "nonlinear"):
            rows.append({
                "protein":            protein,
                "fitness_model":      fm,
                "sequence_length":    L,
                "annotation_density": ann_density,
                "n_annotated":        n_ann,
                "max_category_weight": max_w,
                "has_cliff":          int(has_cliff and fm == "nonlinear"),
            })
    return pd.DataFrame(rows)


def fit_decision_tree(
    auc_df: pd.DataFrame,
    scenario_features: pd.DataFrame,
    output_dir: Path,
    dpi: int = 150,
) -> None:
    """
    Fit a depth-2 decision tree to predict the best method from scenario features.
    Falls back to printing feature importance if sklearn is unavailable.
    """
    # Best method per scenario = highest AUC
    best = (
        auc_df.sort_values("auc_noise", ascending=False)
        .groupby(["protein", "fitness_model"])
        .first()
        .reset_index()[["protein", "fitness_model", "interp"]]
    )
    best.rename(columns={"interp": "best_method"}, inplace=True)

    merged = best.merge(scenario_features, on=["protein", "fitness_model"], how="inner")
    if merged.empty:
        print("  [Decision tree] No data after merge. Skipping.")
        return

    feature_cols = ["sequence_length", "annotation_density",
                    "n_annotated", "max_category_weight", "has_cliff"]
    feature_cols = [c for c in feature_cols if c in merged.columns]

    methods = sorted(merged["best_method"].unique())
    y = merged["best_method"].map({m: i for i, m in enumerate(methods)}).values
    X = merged[feature_cols].values

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X, y)

        print("\n  [Decision tree] Depth-2 tree predicting best method:")
        tree_txt = export_text(clf, feature_names=feature_cols)
        print(tree_txt)

        # Save text
        tree_txt_path = output_dir / "decision_tree.txt"
        tree_txt_path.write_text(
            "Best-method decision tree (depth 2)\n"
            "=====================================\n"
            f"Target classes: {methods}\n\n"
            + tree_txt
        )

        # Plot tree
        fig, ax = plt.subplots(figsize=(12, 5), facecolor=FIG_BG)
        plot_tree(
            clf, feature_names=feature_cols,
            class_names=[INTERP_LABELS.get(m, m) for m in methods],
            filled=True, rounded=True, fontsize=9, ax=ax,
        )
        ax.set_title("Decision tree: best interpolation method by scenario",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "decision_tree.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: decision_tree.png")

        # Print feature importances
        print("\n  Feature importances:")
        for feat, imp in sorted(zip(feature_cols, clf.feature_importances_),
                                 key=lambda x: -x[1]):
            print(f"    {feat:>25s}: {imp:.3f}")

    except ImportError:
        print("  [Decision tree] scikit-learn not available. "
              "Printing best method per scenario only.")

    # Always print the best-method table
    print("\n  Best method per scenario:")
    print(merged[["protein", "fitness_model", "best_method"]].to_string(index=False))
    merged[["protein", "fitness_model", "best_method"] + feature_cols].to_csv(
        output_dir / "best_method_per_scenario.csv", index=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_composite_heatmap(
    auc_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: scenarios × methods, colour = AUC-over-noise composite score."""
    pivot = auc_df.pivot_table(
        index=["protein", "fitness_model"],
        columns="interp",
        values="auc_noise",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1] * 1.4), max(3, pivot.shape[0] * 0.6 + 1)),
                           facecolor=FIG_BG)
    import matplotlib.cm as cm
    cmap = cm.get_cmap("RdYlGn")
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([INTERP_LABELS.get(c, c) for c in pivot.columns],
                       rotation=30, ha="right", fontsize=8)
    row_labels = [f"{p}/{fm}" for p, fm in pivot.index]
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            txt = f"{val:.2f}" if np.isfinite(val) else "N/A"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < val < 0.85 else "white")
    plt.colorbar(im, ax=ax, label="AUC over noise (composite score)")
    ax.set_title("Composite attribution quality: AUC over noise levels",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_noise_curves(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 150,
) -> None:
    """
    One subplot per scenario (protein × fitness_model), composite score vs noise.
    Each interpolation method is a separate line.
    """
    scenarios = df[["protein", "fitness_model"]].drop_duplicates().values.tolist()
    n = len(scenarios)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3),
                             facecolor=FIG_BG, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (protein, fm) in enumerate(scenarios):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_visible(True)
        ax.set_facecolor(FIG_BG)
        ax.spines[["top", "right"]].set_visible(False)

        sub = df[(df["protein"] == protein) & (df["fitness_model"] == fm)]
        for interp, grp in sub.groupby("interp"):
            grp = grp.sort_values("noise")
            ax.plot(
                grp["noise"], grp["composite"],
                color=INTERP_COLORS.get(interp, "#607D8B"),
                marker=INTERP_MARKERS.get(interp, "o"),
                label=INTERP_LABELS.get(interp, interp),
                linewidth=1.8, markersize=5,
            )
        ax.set_title(f"{protein} / {fm}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Noise level", fontsize=8)
        ax.set_ylabel("Composite score", fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)),
               fontsize=8, framealpha=0, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Composite attribution quality vs noise level",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = output_dir / "noise_curves.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_peak_vs_decay(
    rob_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Scatter: peak_perf (x) vs decay_rate (y), one point per (method, scenario).
    Each method gets a distinct colour / marker.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=FIG_BG)
    scenarios = rob_df[["protein", "fitness_model"]].drop_duplicates()
    scenario_list = [f"{r.protein}/{r.fitness_model}" for r in scenarios.itertuples()]

    for ax_idx, (ax, y_col, y_label) in enumerate(zip(
        axes,
        ["decay_rate", "floor_perf"],
        ["Decay rate  b  (exp fit, higher = faster decay)", "Floor performance at max noise"],
    )):
        ax.set_facecolor(FIG_BG)
        ax.spines[["top", "right"]].set_visible(False)
        for interp, grp in rob_df.groupby("interp"):
            valid = grp.dropna(subset=["peak_perf", y_col])
            ax.scatter(
                valid["peak_perf"], valid[y_col],
                color=INTERP_COLORS.get(interp, "#607D8B"),
                marker=INTERP_MARKERS.get(interp, "o"),
                label=INTERP_LABELS.get(interp, interp),
                s=70, zorder=3, alpha=0.85,
            )
        ax.set_xlabel("Peak performance (composite @ min noise)", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(
            "Peak vs. Decay rate" if ax_idx == 0 else "Peak vs. Floor",
            fontsize=10, fontweight="bold",
        )

    axes[1].legend(loc="best", fontsize=8, framealpha=0)
    fig.suptitle("Noise robustness: peak performance vs decay characteristics",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_rank_by_scenario(
    rank_mat: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Strip plot: average rank per method, one dot per scenario to show variability.
    """
    long_rows = []
    for (protein, fm), row in rank_mat.iterrows():
        for interp, rk in row.items():
            long_rows.append({
                "interp": interp,
                "scenario": f"{protein}/{fm}",
                "rank": rk,
            })
    long = pd.DataFrame(long_rows)
    methods = rank_mat.mean(axis=0).sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 1.5), 4), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.spines[["top", "right"]].set_visible(False)

    for i, m in enumerate(methods):
        sub = long[long["interp"] == m]
        col = INTERP_COLORS.get(m, "#607D8B")
        # jitter
        xs = np.full(len(sub), i) + np.random.uniform(-0.15, 0.15, len(sub))
        ax.scatter(xs, sub["rank"].values, color=col, alpha=0.7, s=50, zorder=3)
        ax.plot([i - 0.3, i + 0.3],
                [sub["rank"].mean(), sub["rank"].mean()],
                color=col, linewidth=2.5, zorder=4)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([INTERP_LABELS.get(m, m) for m in methods],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Rank (1 = best)", fontsize=9)
    ax.set_ylim(0.5, rank_mat.shape[1] + 0.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.invert_yaxis()
    ax.set_title("Method ranks across scenarios  (bar = mean, dots = individual scenarios)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(
    friedman_res: dict,
    W: float,
    rob_df: pd.DataFrame,
    pmat: pd.DataFrame | None,
    alpha: float = 0.05,
) -> None:
    rank_df = friedman_res.get("rank_df")
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"\n  Friedman χ²  = {friedman_res['stat']:.3f}")
    print(f"  p-value      = {friedman_res['pvalue']:.4f}  "
          f"({'SIGNIFICANT' if friedman_res['pvalue'] < alpha else 'not significant'} at α={alpha})")
    print(f"\n  Kendall's W  = {W:.3f}  "
          f"({'High concordance' if W >= 0.7 else 'Low concordance — scenario-specific advice'})")

    if rank_df is not None:
        print("\n  Average method ranks (1 = best):")
        for _, r in rank_df.iterrows():
            lbl = INTERP_LABELS.get(r["method"], r["method"])
            print(f"    {lbl:>20s}: rank {r['avg_rank']:.2f}")

    print("\n  AUC-over-noise summary per method:")
    agg = rob_df.groupby("interp")["auc_noise"].agg(["mean", "std"])
    agg = agg.sort_values("mean", ascending=False)
    for interp, row in agg.iterrows():
        lbl = INTERP_LABELS.get(interp, interp)
        print(f"    {lbl:>20s}:  mean={row['mean']:.3f}  std={row['std']:.3f}")

    if pmat is not None:
        print(f"\n  Nemenyi p-values (bold = p < {alpha}):")
        methods = list(pmat.index)
        header = "  " + " " * 22 + "  ".join(
            [f"{INTERP_LABELS.get(m, m)[:8]:>8s}" for m in methods]
        )
        print(header)
        for m_i in methods:
            row_str = f"  {INTERP_LABELS.get(m_i, m_i):>22s}"
            for m_j in methods:
                p = pmat.loc[m_i, m_j]
                marker = "*" if p < alpha and m_i != m_j else " "
                row_str += f"  {p:>7.3f}{marker}"
            print(row_str)
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method-scenario interaction analysis for IGLOO IG quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir",  default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--proteins",      nargs="+", default=None, choices=_KNOWN_PROTEINS)
    parser.add_argument("--fitness-models", nargs="+", default=None,
                        choices=["additive", "epistatic", "nonlinear"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        metavar="METRIC",
                        help=f"Metrics for composite score (default: {DEFAULT_METRICS})")
    parser.add_argument("--concordance-threshold", type=float, default=0.7)
    parser.add_argument("--multi-seed", action="store_true", default=False)
    parser.add_argument("--min-output-diff", type=float, default=NEAR_ZERO_THRESHOLD)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for Friedman/Nemenyi (default: 0.05)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    output_dir   = Path(args.output_dir)

    if not results_root.exists():
        print(f"[ERROR] Results directory does not exist: {results_root}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nScanning: {results_root}")

    # ── Step 1: Load metrics ─────────────────────────────────────────────────
    print("\n[Step 1] Loading metrics...")
    df = load_metrics(
        results_root,
        proteins=args.proteins,
        fitness_models=args.fitness_models,
        min_output_diff=args.min_output_diff,
        min_count=args.min_count,
        multi_seed=args.multi_seed,
    )
    if df.empty:
        print("No data found. Check --results-dir and filters.")
        sys.exit(0)

    print(f"\n  Loaded {len(df)} rows covering:")
    print(f"    Proteins:        {sorted(df['protein'].unique())}")
    print(f"    Fitness models:  {sorted(df['fitness_model'].unique())}")
    print(f"    Noise levels:    {sorted(df['noise'].unique())}")
    print(f"    Methods:         {sorted(df['interp'].unique())}")

    # ── Step 2: Composite score ──────────────────────────────────────────────
    print("\n[Step 2] Computing composite scores...")
    df = compute_composite_scores(df, args.metrics)
    df.to_csv(output_dir / "composite_scores.csv", index=False)
    print(f"  Saved: composite_scores.csv")

    # ── Step 3: Noise robustness ─────────────────────────────────────────────
    print("\n[Step 3] Computing noise robustness (AUC over noise, decay fit)...")
    rob_df = noise_robustness(df)
    rob_df.to_csv(output_dir / "noise_robustness.csv", index=False)
    print(f"  Saved: noise_robustness.csv")

    # ── Step 4: Friedman test ────────────────────────────────────────────────
    print("\n[Step 4] Friedman test...")
    friedman_res = run_friedman(rob_df)
    print(f"  Friedman χ² = {friedman_res['stat']:.3f},  p = {friedman_res['pvalue']:.4f}")

    n_scenarios = (
        len(friedman_res["pivot"]) if friedman_res.get("pivot") is not None
        else len(rob_df[["protein", "fitness_model"]].drop_duplicates())
    )

    # ── Step 5: Nemenyi post-hoc ─────────────────────────────────────────────
    pmat = None
    if friedman_res.get("rank_mat") is not None:
        print("\n[Step 5] Nemenyi post-hoc...")
        pmat = nemenyi_posthoc(friedman_res["rank_mat"])
        pmat.to_csv(output_dir / "nemenyi_pvalues.csv")
        print(f"  Saved: nemenyi_pvalues.csv")

    # ── Step 6: CD diagram ───────────────────────────────────────────────────
    if friedman_res.get("rank_df") is not None and friedman_res.get("rank_mat") is not None:
        print("\n[Step 6] Critical Difference diagram...")
        plot_cd_diagram(
            friedman_res["rank_df"],
            pmat,
            n_scenarios=n_scenarios,
            alpha=args.alpha,
            output_path=output_dir / "cd_diagram.png",
            dpi=args.dpi,
        )

    # ── Step 7: Kendall's W ──────────────────────────────────────────────────
    W = float("nan")
    if friedman_res.get("rank_mat") is not None:
        print("\n[Step 7] Kendall's W (concordance)...")
        W = kendalls_w(friedman_res["rank_mat"])
        print(f"  Kendall's W = {W:.3f}")
        if W >= args.concordance_threshold:
            print("  → High concordance: a single universal ranking applies.")
        else:
            print("  → Low concordance: method ranking is scenario-dependent.")

    # ── Step 8: Decision tree (if low concordance) ───────────────────────────
    print("\n[Step 8] Scenario features + decision tree...")
    scen_feat = build_scenario_features(
        args.proteins or _KNOWN_PROTEINS
    )
    scen_feat.to_csv(output_dir / "scenario_features.csv", index=False)
    fit_decision_tree(rob_df, scen_feat, output_dir, dpi=args.dpi)

    # ── Visualisations ───────────────────────────────────────────────────────
    print("\n[Figures]")
    plot_noise_curves(df, output_dir, dpi=args.dpi)
    plot_composite_heatmap(rob_df, output_dir / "composite_heatmap.png", dpi=args.dpi)
    plot_peak_vs_decay(rob_df, output_dir / "peak_vs_decay.png", dpi=args.dpi)
    if friedman_res.get("rank_mat") is not None:
        plot_rank_by_scenario(
            friedman_res["rank_mat"],
            output_dir / "rank_by_scenario.png",
            dpi=args.dpi,
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(friedman_res, W, rob_df, pmat, alpha=args.alpha)

    print(f"\nAll outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
