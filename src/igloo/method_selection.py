"""
Method-scenario interaction analysis for IGLOO IG attributions.

Provides the full analysis and plotting pipeline for answering:
"Which interpolation method works best for which protein / fitness-landscape?"

Public API
----------
Analysis
    load_all_metrics(results_root, proteins, fitness_models,
                     min_output_diff, min_count, split_filter)  → pd.DataFrame
    add_composite_score(df)                                      → pd.DataFrame
    build_auc_matrix(df)                                         → pd.DataFrame
    run_friedman(auc_matrix, alpha)                              → dict
    kendalls_w(auc_matrix)                                       → float
    nemenyi_posthoc(auc_matrix)                                  → pd.DataFrame | None
    fit_noise_decay(df)                                          → pd.DataFrame
    build_scenario_feature_df(auc_matrix)                        → pd.DataFrame
    run_decision_tree(auc_matrix, output_path, dpi)              → str | None

Plotting
    plot_critical_difference(auc_matrix, nemenyi_pvals,
                             output_path, alpha, dpi)
    plot_composite_noise_curves(df, output_path, dpi)
    plot_auc_heatmap(auc_matrix, output_path, dpi)
    plot_peak_vs_decay(decay_df, output_path, dpi)
    plot_metric_breakdown(df, output_path, dpi)

Reporting
    print_summary(auc_matrix, friedman_res, W, decay_df, threshold)
    save_report(output_dir, auc_matrix, friedman_res, W, decay_df,
                nemenyi_pvals, tree_rules, threshold)

Constants
    COMPOSITE_METRICS  – list of (metric_key, label) tuples
    METRIC_KEYS        – ordered list of metric keys
    METRIC_LABELS      – {key: label} dict
    SCENARIO_FEATURES  – built-in protein feature lookup table
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import friedmanchisquare, rankdata

# ── project root ────────────────────────────────────────────────────────────
_PKG_DIR      = Path(__file__).resolve().parent   # src/igloo/
_PROJECT_ROOT = _PKG_DIR.parent.parent             # project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.ig_metrics import (                 # noqa: E402
    NEAR_ZERO_THRESHOLD,
    discover_result_dirs,
    compute_metrics_for_dir,
)

# ── optional dependencies ────────────────────────────────────────────────────
try:
    import scikit_posthocs as _sp
    _HAS_SP = True
except ImportError:
    _HAS_SP = False

try:
    from sklearn.tree import DecisionTreeClassifier, export_text as _export_text
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ── method colour / label maps (from plot module, with fallback) ─────────────
try:
    from src.igloo.plot import INTERP_COLORS, INTERP_LABELS  # type: ignore
except Exception:
    INTERP_COLORS: dict[str, str] = {}
    INTERP_LABELS: dict[str, str] = {}

_FALLBACK_COLORS: list[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def _method_color(method: str, palette: list[str] = _FALLBACK_COLORS) -> str:
    return INTERP_COLORS.get(method, palette[hash(method) % len(palette)])


def _method_label(method: str) -> str:
    return INTERP_LABELS.get(method, method)


# ---------------------------------------------------------------------------
# Composite-score configuration
# (abs-IG track; NDCG + AP@R as primary ranking metrics)
# ---------------------------------------------------------------------------
COMPOSITE_METRICS: list[tuple[str, str]] = [
    ("ndcg",          "NDCG"),
    ("ap_graded",     "AP@R (graded)"),
    ("auroc",         "AUROC"),
    ("pv_ndcg_mean",  "Per-variant NDCG"),
    ("sign_acc",      "Sign accuracy"),
    ("cat_sep",       "Category separation"),
    ("mutation_ndcg", "Mutation NDCG"),
]
METRIC_KEYS:   list[str]       = [k for k, _ in COMPOSITE_METRICS]
METRIC_LABELS: dict[str, str]  = {k: lbl for k, lbl in COMPOSITE_METRICS}

# ---------------------------------------------------------------------------
# Per-protein scenario features (used by the decision tree)
# Derived from the synthetic_data PROTEINS registry when available;
# hard-coded fallback values kept for environments without that module.
# ---------------------------------------------------------------------------
try:
    from src.igloo.synthetic_data import PROTEINS as _PROTEIN_DEFS_FOR_FEATURES  # type: ignore
    SCENARIO_FEATURES: dict[str, dict] = {
        name: {
            "sequence_length":     len(p.sequence),
            "annotation_density":  len(p.annotations) / max(len(p.sequence), 1),
            "n_annotated":         len(p.annotations),
            "max_category_weight": max(p.category_weights.values()) if p.category_weights else 1.0,
            "has_cliff":           int(any(
                cat in {"active_site", "metal_binding"}
                for cat in p.annotations.values()
            )),
        }
        for name, p in _PROTEIN_DEFS_FOR_FEATURES.items()
    }
except Exception:
    SCENARIO_FEATURES = {
        "GB1":        {"sequence_length": 56,  "annotation_density": 51 / 56,
                       "n_annotated": 51, "max_category_weight": 1.5,  "has_cliff": 0},
        "T4Lysozyme": {"sequence_length": 164, "annotation_density": 7 / 164,
                       "n_annotated": 7,  "max_category_weight": 5.0,  "has_cliff": 1},
        "p53":        {"sequence_length": 393, "annotation_density": 45 / 393,
                       "n_annotated": 45, "max_category_weight": 5.0,  "has_cliff": 1},
    }


# ===========================================================================
# Step 1 — Load raw metrics
# ===========================================================================

def load_all_metrics(
    results_root: Path,
    proteins: list[str] | None = None,
    fitness_models: list[str] | None = None,
    min_output_diff: float = NEAR_ZERO_THRESHOLD,
    min_count: int = 5,
    split_filter: str | None = None,
) -> pd.DataFrame:
    """Return a long-form DataFrame with one row per (protein, fm, noise, method).

    Each row contains the raw values for every key in ``METRIC_KEYS``.

    Parameters
    ----------
    results_root :
        Root directory of the synthetic results tree.
    proteins :
        Restrict to specific proteins (``None`` = all).
    fitness_models :
        Restrict to specific fitness models (``None`` = all).
    min_output_diff :
        Passed to ``compute_metrics_for_dir``.
    min_count :
        Passed to ``compute_metrics_for_dir``.
    split_filter :
        ``'trainval'``, ``'test'``, or ``None`` (all sequences).
    """
    entries = discover_result_dirs(
        results_root,
        proteins=proteins,
        fitness_models=fitness_models,
    )
    if not entries:
        raise FileNotFoundError(
            f"No result directories found under {results_root}. "
            "Check --results-dir and protein / fitness-model filters."
        )

    rows: list[dict] = []
    for protein, fm, noise, interp, path in sorted(entries):
        label = f"{protein}/{fm}/noise={noise}/{interp}"
        print(f"  Loading {label} ... ", end="", flush=True)
        metrics = compute_metrics_for_dir(
            path, protein,
            min_output_diff=min_output_diff,
            min_count=min_count,
            fitness_model=fm,
            split_filter=split_filter,
        )
        if metrics is None:
            print("SKIP")
            continue
        row: dict = {
            "protein":       protein,
            "fitness_model": fm,
            "noise":         noise,
            "method":        interp,
            "scenario":      f"{protein}_{fm}",
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, float("nan"))
        row["collapse_frac"] = metrics.get("collapse_frac", float("nan"))
        row["n_reliable"]    = metrics.get("n_reliable",    float("nan"))
        rows.append(row)
        vals = "  ".join(
            f"{METRIC_LABELS[k]}={row[k]:.3f}" for k in METRIC_KEYS
            if not np.isnan(row[k])
        )
        coll = row["collapse_frac"]
        n_rel = int(row["n_reliable"]) if not np.isnan(row["n_reliable"]) else "?"
        print(f"ok  [{vals}]  collapse={coll:.3f}  n_reliable={n_rel}")

    if not rows:
        raise RuntimeError("No metrics could be computed for any result directory.")

    return pd.DataFrame(rows)


# ===========================================================================
# Step 2 — Composite score
# ===========================================================================

def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise each metric within (scenario, noise), then average.

    Adds one ``_norm_{key}`` column per metric and a ``composite`` column.
    All transforms are applied per (scenario, noise) block so that scores
    from different scenarios remain comparable in a rank sense.
    """
    df = df.copy()
    normed_cols: list[str] = []
    for key in METRIC_KEYS:
        col = f"_norm_{key}"
        normed_cols.append(col)
        df[col] = float("nan")
        for (scenario, noise), grp in df.groupby(["scenario", "noise"]):
            idx  = grp.index
            vals = grp[key].values.astype(float)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax > vmin:
                df.loc[idx, col] = (vals - vmin) / (vmax - vmin)
            else:
                df.loc[idx, col] = 0.5  # all methods tied on this metric
    df["composite"] = df[normed_cols].mean(axis=1)
    return df


# ===========================================================================
# Step 3 — AUC over noise → (scenario × method) matrix
# ===========================================================================

def build_auc_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Trapezoidal AUC of composite score over noise, per (method, scenario).

    Returns a DataFrame indexed by scenario, with methods as columns.
    AUC is normalised by the noise range so it lies in [0, 1] regardless
    of the number or spacing of noise levels.
    """
    records: list[dict] = []
    for (method, scenario), grp in df.groupby(["method", "scenario"]):
        grp_s = grp.sort_values("noise")
        x = grp_s["noise"].values.astype(float)
        y = grp_s["composite"].values.astype(float)
        if len(x) >= 2 and x[-1] > x[0]:
            auc = float(np.trapz(y, x) / (x[-1] - x[0]))
        else:
            auc = float(np.nanmean(y))
        records.append({"method": method, "scenario": scenario, "auc": auc})

    return (
        pd.DataFrame(records)
        .pivot(index="scenario", columns="method", values="auc")
    )


# ===========================================================================
# Step 4 — Friedman test
# ===========================================================================

def run_friedman(auc_matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Non-parametric test for differences across interpolation methods.

    Uses the AUC matrix; scenarios are the blocks.  Drops any scenario row
    with a missing value for any method before testing.

    Returns a dict with keys ``stat``, ``p``, ``significant``,
    ``n_scenarios``, ``n_methods``.
    """
    mask  = ~np.any(np.isnan(auc_matrix.values), axis=1)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        dropped = auc_matrix.index[~mask].tolist()
        print(f"  WARNING: Friedman test dropping {n_dropped} incomplete scenario(s): {dropped}")
    clean = auc_matrix.loc[mask]
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return {"stat": float("nan"), "p": float("nan"), "significant": False,
                "n_scenarios": int(mask.sum()), "n_methods": auc_matrix.shape[1]}
    arrays = [clean[c].values for c in clean.columns]
    stat, p = friedmanchisquare(*arrays)
    return {
        "stat":        float(stat),
        "p":           float(p),
        "significant": bool(p < alpha),
        "n_scenarios": int(clean.shape[0]),
        "n_methods":   int(clean.shape[1]),
    }


# ===========================================================================
# Step 5a — Kendall's W
# ===========================================================================

def kendalls_w(auc_matrix: pd.DataFrame) -> float:
    """Tie-corrected Kendall's W across scenarios.

    Scenarios are the raters, methods are the objects being ranked.
    W = 1 → all scenarios agree on the method ranking.
    W = 0 → no agreement.

    Uses the tie-corrected denominator because min-max normalisation within
    small method groups frequently produces tied composite scores.
    """
    mask  = ~np.any(np.isnan(auc_matrix.values), axis=1)
    clean = auc_matrix.loc[mask].values    # (n_scenarios, n_methods)
    n, k  = clean.shape
    if n < 2 or k < 2:
        return float("nan")
    ranks = np.apply_along_axis(lambda r: rankdata(-r), axis=1, arr=clean)
    R     = ranks.sum(axis=0)
    S     = np.sum((R - R.mean()) ** 2)
    # Tie correction: T = Σ (t³ − t) summed over all tie groups in each rater
    T = 0.0
    for i in range(n):
        _, counts = np.unique(ranks[i], return_counts=True)
        T += float(np.sum(counts ** 3 - counts))
    denom = n ** 2 * (k ** 3 - k) - n * T
    return float(12 * S / denom) if denom > 0 else float("nan")


# ===========================================================================
# Step 5b — Nemenyi post-hoc
# ===========================================================================

def nemenyi_posthoc(auc_matrix: pd.DataFrame) -> pd.DataFrame | None:
    """Pairwise Nemenyi post-hoc test on the AUC matrix.

    Requires ``scikit-posthocs``.  Returns a square p-value DataFrame with
    method names as both index and columns, or ``None`` if the package is
    not installed.
    """
    if not _HAS_SP:
        print("  [scikit-posthocs not installed — Nemenyi test skipped]")
        return None
    mask    = ~np.any(np.isnan(auc_matrix.values), axis=1)
    clean   = auc_matrix.loc[mask]
    methods = list(clean.columns)
    pv      = _sp.posthoc_nemenyi_friedman(clean.values)
    # Label rows and columns with method names for safe loc-based indexing
    pv.index   = methods
    pv.columns = methods
    return pv


# ===========================================================================
# Step 6 — Noise-curve decay characterisation
# ===========================================================================

def _exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay model: score(noise) ≈ a · exp(−b · noise) + c."""
    return a * np.exp(-b * x) + c


def fit_noise_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the exponential decay model per (method, scenario).

    Returns a DataFrame with columns:
        method, scenario, peak, decay_rate, floor, fit_r2

    ``peak``       – fitted value at noise = 0 (= a + c)
    ``decay_rate`` – fitted b  (higher = faster decay, less robust)
    ``floor``      – fitted c  (asymptotic performance at high noise)
    ``fit_r2``     – R² of the fit (NaN if fewer than 3 data points)
    """
    records: list[dict] = []
    for (method, scenario), grp in df.groupby(["method", "scenario"]):
        grp_s = grp.sort_values("noise")
        x = grp_s["noise"].values.astype(float)
        y = grp_s["composite"].values.astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        # Fallback peak/floor: use the observed value at lowest/highest noise
        _noise_min_idx = int(np.argmin(x)) if len(x) > 0 else 0
        _noise_max_idx = int(np.argmax(x)) if len(x) > 0 else 0
        peak       = float(y[_noise_min_idx]) if len(y) > 0 else float("nan")
        floor      = float(y[_noise_max_idx]) if len(y) > 0 else float("nan")
        fit_r2     = float("nan")
        decay_rate = float("nan")

        if len(x) >= 3:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p0 = (max(y) - min(y), 1.0, min(y))
                    popt, _ = curve_fit(
                        _exp_decay, x, y, p0=p0,
                        bounds=([0, 0, -1], [2, 20, 2]),
                        maxfev=5000,
                    )
                a_f, b_f, c_f = popt
                y_pred     = _exp_decay(x, *popt)
                ss_res     = np.sum((y - y_pred) ** 2)
                ss_tot     = np.sum((y - y.mean()) ** 2)
                fit_r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                decay_rate = float(b_f)
                peak       = float(a_f + c_f)   # model peak at noise = 0
                floor      = float(c_f)
            except Exception:
                pass  # keep obs-based fallbacks

        records.append({
            "method":      method,
            "scenario":    scenario,
            "peak":        peak,
            "decay_rate":  decay_rate,
            "floor":       floor,
            "fit_r2":      fit_r2,
        })
    return pd.DataFrame(records)


# ===========================================================================
# Decision tree (conditional recommendation)
# ===========================================================================

def build_scenario_feature_df(auc_matrix: pd.DataFrame) -> pd.DataFrame:
    """Map each scenario in *auc_matrix* to a fixed feature vector.

    Uses ``SCENARIO_FEATURES`` for protein-level properties.
    Fitness-model columns ``is_epistatic`` / ``is_nonlinear`` are binary.
    """
    rows: list[dict] = []
    for scenario in auc_matrix.index:
        parts   = scenario.split("_", 1)
        protein = parts[0]
        fm      = parts[1] if len(parts) > 1 else "additive"
        base    = SCENARIO_FEATURES.get(protein, {})
        rows.append({
            "scenario":            scenario,
            "protein":             protein,
            "fitness_model":       fm,
            "sequence_length":     base.get("sequence_length",     100),
            "annotation_density":  base.get("annotation_density",  0.05),
            "n_annotated":         base.get("n_annotated",         5),
            "max_category_weight": base.get("max_category_weight", 2.0),
            "has_cliff":           base.get("has_cliff",           0),
            "is_epistatic":        int(fm == "epistatic"),
            "is_nonlinear":        int(fm == "nonlinear"),
        })
    return pd.DataFrame(rows).set_index("scenario")


def run_decision_tree(
    auc_matrix: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> str | None:
    """Fit a depth-2 decision tree predicting the best method per scenario.

    Saves a feature-importance bar chart to *output_path*.
    Returns the text representation of the tree rules, or ``None`` if
    scikit-learn is not available or there are too few scenarios.
    """
    if not _HAS_SKLEARN:
        print("  [scikit-learn not installed — decision tree skipped]")
        return None

    feat_df  = build_scenario_feature_df(auc_matrix)
    best     = auc_matrix.idxmax(axis=1)
    common   = auc_matrix.index.intersection(feat_df.index)
    if len(common) < 3:
        print("  [Too few scenarios for a decision tree]")
        return None

    feat_cols = [
        "sequence_length", "annotation_density", "n_annotated",
        "max_category_weight", "has_cliff", "is_epistatic", "is_nonlinear",
    ]
    X = feat_df.loc[common, feat_cols].values
    y = best.loc[common].values

    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    rules = _export_text(clf, feature_names=feat_cols)

    # Feature-importance bar chart
    importances = clf.feature_importances_
    fig, ax     = plt.subplots(figsize=(7, 3.5))
    cols        = [_FALLBACK_COLORS[i % len(_FALLBACK_COLORS)] for i in range(len(feat_cols))]
    bars        = ax.barh(feat_cols, importances, color=cols)
    ax.set_xlabel("Feature importance", fontsize=11)
    ax.set_title("Decision tree: scenario features predicting the best method", fontsize=11)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return rules


# ===========================================================================
# Plotting
# ===========================================================================

def plot_critical_difference(
    auc_matrix: pd.DataFrame,
    nemenyi_pvals: pd.DataFrame | None,
    output_path: Path,
    alpha: float = 0.05,
    dpi: int = 150,
) -> None:
    """Critical-difference diagram (Demšar 2006) for the AUC matrix.

    Methods are plotted by average rank (1 = best, left).
    Horizontal bars connect groups whose pairwise Nemenyi p > *alpha*
    (i.e. not significantly different from each other).
    """
    methods = list(auc_matrix.columns)
    mask    = ~np.any(np.isnan(auc_matrix.values), axis=1)
    clean   = auc_matrix.loc[mask].values
    n, k    = clean.shape

    ranks_per_scenario = np.apply_along_axis(
        lambda r: rankdata(-r), axis=1, arr=clean
    )
    avg_ranks = ranks_per_scenario.mean(axis=0)
    order     = np.argsort(avg_ranks)    # best → worst

    # q_α (two-tailed, Demšar Table 5)
    Q_ALPHA   = {0.10: 2.291, 0.05: 2.569, 0.01: 3.143}
    q         = Q_ALPHA.get(alpha, 2.569)
    cd        = q * np.sqrt(k * (k + 1) / (6 * n))

    fig, ax   = plt.subplots(figsize=(max(8, k * 1.5), 3.5))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.5, 1.0)
    ax.set_xlabel("Average rank across scenarios (1 = best)", fontsize=11)
    ax.set_yticks([])
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))

    for i, idx in enumerate(order):
        r   = avg_ranks[idx]
        lbl = _method_label(methods[idx])
        col = _method_color(methods[idx])
        ax.plot(r, 0, "o", color=col, ms=8, zorder=3)
        ax.text(r, 0.15 + 0.25 * (i % 2), lbl,
                ha="center", va="bottom", fontsize=9, color=col)

    # CD reference bar
    ax.annotate("", xy=(1 + cd, -0.9), xytext=(1, -0.9),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1 + cd / 2, -1.15, f"CD={cd:.2f}", ha="center", fontsize=8)

    # Clique bars (not significantly different) — connected components via union-find
    if nemenyi_pvals is not None:
        # Build adjacency: edge (i,j) if methods i,j are NOT significantly different
        parent = list(range(k))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            parent[_find(a)] = _find(b)

        for i in range(k):
            for j in range(i + 1, k):
                m_i = methods[order[i]]
                m_j = methods[order[j]]
                if nemenyi_pvals.loc[m_i, m_j] >= alpha:
                    _union(i, j)

        # Collect connected components
        from collections import defaultdict
        components: dict[int, list[int]] = defaultdict(list)
        for i in range(k):
            components[_find(i)].append(i)

        drawn_count = 0
        for comp in components.values():
            if len(comp) > 1:
                xs    = sorted(avg_ranks[order[g]] for g in comp)
                y_bar = -0.5 - 0.12 * drawn_count
                ax.plot([xs[0], xs[-1]], [y_bar, y_bar],
                        lw=3, color="black", alpha=0.6, solid_capstyle="butt")
                drawn_count += 1

    ax.set_title(
        f"Critical-difference diagram  (α={alpha}, n={n} scenarios)",
        fontsize=12, pad=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_composite_noise_curves(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """One subplot per scenario: composite score vs noise level, one line per method."""
    scenarios = sorted(df["scenario"].unique())
    methods   = sorted(df["method"].unique())
    ncols     = min(3, len(scenarios))
    nrows     = int(np.ceil(len(scenarios) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    for i, scenario in enumerate(scenarios):
        ax  = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        sub = df[df["scenario"] == scenario]
        for method in methods:
            m_sub = sub[sub["method"] == method].sort_values("noise")
            if m_sub.empty:
                continue
            ax.plot(m_sub["noise"], m_sub["composite"],
                    "o-", color=_method_color(method),
                    label=_method_label(method), lw=1.8, ms=5)
        ax.set_title(scenario.replace("_", " / "), fontsize=10)
        ax.set_xlabel("Noise level", fontsize=9)
        ax.set_ylabel("Composite score", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Shared legend outside subplots
    handles = [
        mpatches.Patch(color=_method_color(m), label=_method_label(m))
        for m in methods
    ]
    fig.legend(
        handles=handles, fontsize=8, loc="lower center",
        ncol=min(len(methods), 5),
        bbox_to_anchor=(0.5, -0.02), framealpha=0,
    )

    fig.suptitle(
        "Composite score (NDCG + AP@R + AUROC + pv-NDCG + sign-acc + cat-sep) "
        "vs noise",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_auc_heatmap(
    auc_matrix: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap of AUC-over-noise composite scores (methods as rows, scenarios as columns)."""
    data = auc_matrix.T    # (methods × scenarios)
    fig, ax = plt.subplots(
        figsize=(max(6, len(auc_matrix.index) * 1.1), max(4, len(data.index) * 0.7))
    )
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="AUC-over-noise composite score")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([s.replace("_", "\n") for s in data.columns], fontsize=9)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([_method_label(m) for m in data.index], fontsize=9)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title("AUC over noise — composite score (higher = better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_peak_vs_decay(
    decay_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Scatter: noise decay rate (x) vs peak performance (y), one subplot per scenario.

    Ideal methods appear in the top-left (high peak, slow decay).
    """
    scenarios = sorted(decay_df["scenario"].unique())
    ncols     = min(3, len(scenarios))
    nrows     = int(np.ceil(len(scenarios) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    for i, scenario in enumerate(scenarios):
        ax  = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        sub = decay_df[decay_df["scenario"] == scenario]
        for _, row in sub.iterrows():
            if np.isnan(row["peak"]) or np.isnan(row["decay_rate"]):
                continue
            col = _method_color(row["method"])
            ax.scatter(row["decay_rate"], row["peak"], color=col, s=80, zorder=3)
            ax.text(row["decay_rate"], row["peak"] + 0.02,
                    _method_label(row["method"]),
                    ha="center", va="bottom", fontsize=7, color=col)
        ax.set_title(scenario.replace("_", " / "), fontsize=10)
        ax.set_xlabel("Decay rate b  (higher = less robust)", fontsize=9)
        ax.set_ylabel("Peak performance (noise = 0)", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Peak performance vs noise-decay rate\n"
        "(top-left = best: high peak AND slow decay)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metric_breakdown(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Grouped bar chart: normalised per-metric score at noise = 0, per scenario."""
    methods   = sorted(df["method"].unique())
    scenarios = sorted(df["scenario"].unique())
    df0       = df[df["noise"] == df["noise"].min()].copy()

    fig, axes = plt.subplots(
        len(COMPOSITE_METRICS), 1,
        figsize=(max(8, len(scenarios) * 1.2), 2.8 * len(COMPOSITE_METRICS)),
        squeeze=False,
    )
    x     = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    for row_i, (key, label) in enumerate(COMPOSITE_METRICS):
        ax       = axes[row_i][0]
        norm_col = f"_norm_{key}"
        for j, method in enumerate(methods):
            sub  = df0[df0["method"] == method].set_index("scenario")
            vals = [
                sub.loc[s, norm_col] if s in sub.index and norm_col in sub.columns
                else float("nan")
                for s in scenarios
            ]
            ax.bar(
                x + j * width - 0.4 + width / 2, vals,
                width=width * 0.9,
                color=_method_color(method),
                alpha=0.85,
                label=_method_label(method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
        ax.set_ylabel("Norm. score", fontsize=9)
        ax.set_title(f"{label} (normalised, noise = 0)", fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(axis="y", alpha=0.3)
        if row_i == 0:
            ax.legend(fontsize=8, ncol=len(methods))

    fig.suptitle("Per-metric normalised scores at zero noise", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Reporting
# ===========================================================================

def print_summary(
    auc_matrix: pd.DataFrame,
    friedman_res: dict,
    W: float,
    decay_df: pd.DataFrame,
    threshold: float = 0.6,
) -> None:
    """Print a human-readable summary of all analysis steps to stdout."""
    print("\n" + "=" * 60)
    print("METHOD-SCENARIO INTERACTION SUMMARY")
    print("=" * 60)

    sig = friedman_res.get("significant", False)
    print(f"\nFriedman test:  χ²={friedman_res['stat']:.3f}  "
          f"p={friedman_res['p']:.4f}  "
          f"({'significant' if sig else 'NOT significant'})")
    print(f"Kendall's W   = {W:.3f}  "
          f"({'stable → universal recommendation' if W >= threshold else 'unstable → conditional recommendation'})")

    print("\nAUC-over-noise composite score matrix (higher = better):")
    print(auc_matrix.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nAverage AUC per method (across all scenarios):")
    for m, v in auc_matrix.mean(axis=0).sort_values(ascending=False).items():
        print(f"  {_method_label(m):30s}  {v:.3f}")

    print("\nBest method per scenario:")
    for scenario, method in auc_matrix.idxmax(axis=1).items():
        print(f"  {scenario:30s}  →  {_method_label(method)}")

    if not decay_df.empty:
        print("\nNoise-decay characterisation:")
        cols = ["scenario", "method", "peak", "decay_rate", "floor", "fit_r2"]
        print(
            decay_df[cols]
            .sort_values(["scenario", "peak"], ascending=[True, False])
            .to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
    print("=" * 60)


def save_report(
    output_dir: Path,
    auc_matrix: pd.DataFrame,
    friedman_res: dict,
    W: float,
    decay_df: pd.DataFrame,
    nemenyi_pvals: pd.DataFrame | None = None,
    tree_rules: str | None = None,
    threshold: float = 0.6,
    raw_df: pd.DataFrame | None = None,
) -> None:
    """Write a JSON summary report and CSV tables to *output_dir*."""
    # NOTE: composite AUC values are relative (normalised within each scenario ×
    # noise block), so they reflect rank differences between methods, not absolute
    # attribution quality.  Do not compare values across scenarios.
    report: dict = {
        "friedman":                 friedman_res,
        "kendalls_w":               W,
        "concordance_stable":       W >= threshold,
        "auc_matrix":               auc_matrix.reset_index().to_dict(orient="records"),
        "best_method_per_scenario": auc_matrix.idxmax(axis=1).to_dict(),
        "avg_auc_per_method":       auc_matrix.mean(axis=0)
                                    .sort_values(ascending=False).to_dict(),
    }
    if not decay_df.empty:
        report["decay"] = (
            decay_df[["method", "scenario", "peak", "decay_rate", "floor", "fit_r2"]]
            .to_dict(orient="records")
        )
    if nemenyi_pvals is not None:
        report["nemenyi_pvals"] = nemenyi_pvals.to_dict()
    if tree_rules is not None:
        report["decision_tree_rules"] = tree_rules

    json_path = output_dir / "method_analysis_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"  Saved: {json_path}")

    auc_matrix.to_csv(output_dir / "auc_matrix.csv")
    print(f"  Saved: {output_dir / 'auc_matrix.csv'}")

    if not decay_df.empty:
        decay_df.to_csv(output_dir / "noise_decay.csv", index=False)
        print(f"  Saved: {output_dir / 'noise_decay.csv'}")

    if raw_df is not None and "collapse_frac" in raw_df.columns:
        collapse_cols = ["protein", "fitness_model", "noise", "method",
                         "collapse_frac", "n_reliable"]
        present = [c for c in collapse_cols if c in raw_df.columns]
        cstats = (
            raw_df[present]
            .sort_values(["protein", "fitness_model", "noise", "method"])
        )
        cstats.to_csv(output_dir / "collapse_stats.csv", index=False)
        print(f"  Saved: {output_dir / 'collapse_stats.csv'}")
