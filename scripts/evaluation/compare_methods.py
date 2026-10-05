#!/usr/bin/env python3
"""
compare_methods.py
------------------
Combines attribution-quality metrics (from ig_metrics / plot_ig_quality) with
mutation-coverage analysis (from compare_attribution_coverage) into a single
figure per protein × fitness-model group.

Layout (one figure per group, saved as {protein}_{fm}_method_comparison.png):

  Top panel    — Bar chart of quality metrics (NDCG, AP@R, AUROC, Per-variant
                 NDCG, Sign Accuracy, Category Separation, Mutation NDCG) per
                 method — the same metrics used to compute the composite score.
  Bottom panel — Attribution vs mutation-coverage plot (one row per method):
                 normalised |IG| line overlaid on normalised mutation-frequency
                 fill; red ★ marks uncovered hot-spots; ground-truth regions
                 shaded if --ground-truth is supplied.

Usage
-----
    # GB1 additive/epistatic/nonlinear, noise_0.0, all 5 methods
    python scripts/evaluation/compare_methods.py \\
        --results-dirs \\
            results/synthetic/GB1_additive/noise_0.0/linear \\
            results/synthetic/GB1_additive/noise_0.0/slerp \\
            results/synthetic/GB1_additive/noise_0.0/nearest_neighbor \\
            results/synthetic/GB1_additive/noise_0.0/gradient_x_input \\
            results/synthetic/GB1_additive/noise_0.0/input_difference \\
            results/synthetic/GB1_epistatic/noise_0.0/linear \\
            results/synthetic/GB1_epistatic/noise_0.0/slerp \\
            results/synthetic/GB1_epistatic/noise_0.0/nearest_neighbor \\
            results/synthetic/GB1_epistatic/noise_0.0/gradient_x_input \\
            results/synthetic/GB1_epistatic/noise_0.0/input_difference \\
            results/synthetic/GB1_nonlinear/noise_0.0/linear \\
            results/synthetic/GB1_nonlinear/noise_0.0/slerp \\
            results/synthetic/GB1_nonlinear/noise_0.0/nearest_neighbor \\
            results/synthetic/GB1_nonlinear/noise_0.0/gradient_x_input \\
            results/synthetic/GB1_nonlinear/noise_0.0/input_difference \\
        --proteins GB1 \\
        --output-dir results/synthetic/evaluation/method_comparison

    # Or scan a whole noise-level directory automatically:
    python scripts/evaluation/compare_methods.py \\
        --results-dir-tree results/synthetic/GB1_additive/noise_0.0 \\
        --proteins GB1 \\
        --output-dir results/synthetic/evaluation/method_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.transparent"] = True
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from igloo.ig_metrics import (   # noqa: E402
    compute_metrics_for_dir,
    _build_importance_vec,
    _wt_sequence,
)
from igloo.plot import plot_attribution_vs_mutation_coverage  # noqa: E402

try:
    from igloo.method_selection import (  # noqa: E402
        run_friedman,
        kendalls_w,
        nemenyi_posthoc,
        plot_critical_difference,
    )
    _STATS_AVAILABLE = True
except Exception:
    _STATS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Display order and labels for known methods
_METHOD_ORDER = ["linear", "slerp", "nearest_neighbor", "gradient_x_input", "input_difference"]
_METHOD_LABELS = {
    "linear":           "Linear",
    "slerp":            "SLERP",
    "nearest_neighbor": "Nearest\nNeighbor",
    "gradient_x_input": "Gradient\n×Input",
    "input_difference": "Input\nDiff.",
}
_METHOD_COLORS = {
    "linear":           "#2196F3",
    "slerp":            "#4CAF50",
    "nearest_neighbor": "#FF9800",
    "gradient_x_input": "#9C27B0",
    "input_difference": "#F44336",
}


def _decompose_method(method: str) -> tuple[str, str]:
    """Split 'plm/interp' → ('plm', 'interp'), 'interp' → ('IGLOO', 'interp')."""
    if "/" in method:
        plm, interp = method.split("/", 1)
        return plm, interp
    return "IGLOO", method


_PLM_ORDER   = ["IGLOO", "ankh_base", "e1_150m"]
_PLM_LABELS  = {"IGLOO": "ESM-C", "ankh_base": "Ankh-Base", "e1_150m": "E1-150M"}
_PLM_COLORS  = {"IGLOO": "#455A64", "ankh_base": "#E91E63", "e1_150m": "#FF9800"}
_PLM_HATCHES = {"IGLOO": "", "ankh_base": "///", "e1_150m": "..."}

# Same keys / labels as add_composite_score() in igloo.method_selection
_METRIC_KEYS   = ["ndcg", "ap_graded", "auroc", "pv_ndcg_mean", "sign_acc", "cat_sep", "mutation_ndcg"]
_METRIC_LABELS = {
    "ndcg":         "NDCG",
    "ap_graded":    "AP@R\n(graded)",
    "auroc":        "AUROC",
    "pv_ndcg_mean": "Per-var\nNDCG",
    "sign_acc":     "Sign Acc.",
    "cat_sep":      "Cat. Sep.",
    "mutation_ndcg":"Mut. NDCG",
}


# ---------------------------------------------------------------------------
# Helpers shared with compare_attribution_coverage.py
# ---------------------------------------------------------------------------

def _find_importance_csv(d: Path) -> Path | None:
    hits = sorted(d.glob("*_position_importance_summary.csv"))
    return hits[0] if hits else None


def _discover_leaf_dirs(tree_root: Path, igloo_only: bool = False) -> list[Path]:
    dirs = sorted(
        p.parent for p in tree_root.rglob("*_position_importance_summary.csv")
    )
    if igloo_only:
        dirs = [d for d in dirs if not any(part.startswith("plm_") for part in d.parts)]
    return dirs


def _mutation_frequency(
    df: pd.DataFrame,
    exclude_wt: bool = True,
    mutation_col: str = "mutation_string",
) -> pd.Series:
    if mutation_col not in df.columns:
        for alias in ("mutation_string", "mutant", "mutations", "variant"):
            if alias in df.columns:
                mutation_col = alias
                break
        else:
            raise ValueError(f"Cannot find mutation column. Available: {list(df.columns)}")

    if exclude_wt and "is_wildtype" in df.columns:
        df = df[~df["is_wildtype"].astype(bool)]

    wt_sentinels = {"", "wt", "wildtype", "wild_type", "wild-type"}
    pos_counts: dict[int, int] = {}
    for ms in df[mutation_col].dropna():
        if str(ms).strip().lower() in wt_sentinels or str(ms).strip().upper() == "WT":
            continue
        for mut in ms.split(";"):
            mut = mut.strip()
            if not mut:
                continue
            digits = "".join(c for c in mut if c.isdigit())
            if digits:
                pos = int(digits)
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return pd.Series(pos_counts, dtype=int)


def _infer_seq_length(df: pd.DataFrame, gt: dict | None) -> int:
    if gt and "sequence_length" in gt:
        return int(gt["sequence_length"])
    wt_rows = df[df.get("is_wildtype", pd.Series(False, index=df.index)).astype(bool)]
    seq_col = next((c for c in ["SEQ", "sequence", "Sequence"] if c in df.columns), None)
    if seq_col and not wt_rows.empty:
        return len(wt_rows.iloc[0][seq_col])
    if seq_col:
        return int(df[seq_col].str.len().max())
    raise ValueError(
        "Cannot infer sequence length — supply --ground-truth or ensure the "
        "dataset has a SEQ/sequence column."
    )


def _infer_protein_and_fm(d: Path) -> tuple[str, str] | tuple[None, None]:
    """Infer protein and fitness_model from directory path components."""
    # Walk ancestors looking for {Protein}_{fm} pattern
    known_proteins = ["GB1", "T4Lysozyme", "p53", "PafA", "ERBC"]
    known_fms      = [
        "additive_deep", "epistatic_deep", "nonlinear_deep",
        "additive", "epistatic", "nonlinear",
    ]
    for part in d.parts:
        for prot in known_proteins:
            if part.startswith(prot + "_"):
                remainder = part[len(prot) + 1:]
                for fm in known_fms:
                    if remainder == fm or remainder.startswith(fm + "_"):
                        return prot, fm
    return None, None


# ---------------------------------------------------------------------------
# Metrics bar chart
# ---------------------------------------------------------------------------

def _plot_metrics_grid(
    axes: list[plt.Axes],
    methods: list[str],
    metrics_by_method: dict[str, dict[str, Any]],
) -> None:
    """Draw one subplot per metric; x=interp method, grouped bars=pLM."""
    # Decompose methods into (plm, interp) pairs
    decomposed = [_decompose_method(m) for m in methods]

    # Ordered unique interp methods and plms present in the data
    seen_interp: list[str] = []
    seen_plm: list[str] = []
    for plm, interp in decomposed:
        if interp not in seen_interp:
            seen_interp.append(interp)
        if plm not in seen_plm:
            seen_plm.append(plm)
    # Respect canonical orders if all entries are known
    interp_order = [m for m in _METHOD_ORDER if m in seen_interp] + \
                   [m for m in seen_interp if m not in _METHOD_ORDER]
    plm_order    = [p for p in _PLM_ORDER if p in seen_plm] + \
                   [p for p in seen_plm if p not in _PLM_ORDER]

    n_interp = len(interp_order)
    n_plm    = len(plm_order)
    x        = np.arange(n_interp)
    width    = 0.8 / max(n_plm, 1)

    for ax_idx, mk in enumerate(_METRIC_KEYS):
        ax = axes[ax_idx]
        for pi, plm in enumerate(plm_order):
            # Find method string matching this (plm, interp) pair
            values = []
            for interp in interp_order:
                full = f"{plm}/{interp}" if plm != "IGLOO" else interp
                m = metrics_by_method.get(full)
                if m and mk in m and m[mk] == m[mk]:
                    values.append(float(m[mk]))
                else:
                    values.append(float("nan"))
            offsets = x + (pi - n_plm / 2 + 0.5) * width
            color   = _PLM_COLORS.get(plm, "#888888")
            hatch   = _PLM_HATCHES.get(plm, "")
            label   = _PLM_LABELS.get(plm, plm) if ax_idx == 0 else "_nolegend_"
            ax.bar(offsets, values, width * 0.88, color=color, hatch=hatch,
                   alpha=0.85, label=label, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_METHOD_LABELS.get(m, m).replace("\n", "\n") for m in interp_order],
            fontsize=7, rotation=40, ha="right", rotation_mode="anchor",
        )
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(_METRIC_LABELS[mk], fontsize=8, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Score", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Legend on first axis
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.85,
                   handlelength=1.2, handletextpad=0.4)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def _make_comparison_figure(
    protein: str,
    fm: str,
    methods: list[str],
    metrics_by_method: dict[str, dict | None],
    importance_by_method: dict[str, pd.DataFrame],
    mut_freq: pd.Series,
    seq_length: int,
    gt: dict | None,
    uncovered_threshold_pct: float,
    output_dir: Path,
    dpi: int,
    title_suffix: str = "",
) -> Path:
    """Build and save two figures for one protein × fitness-model group:
    1. ``*_metrics.png``  — bar-chart grid (one panel per quality metric)
    2. ``*_coverage.png`` — attribution-vs-coverage signal panels
    Returns the path to the metrics figure.
    """
    base = f"{protein}_{fm}_method_comparison"
    if title_suffix:
        base += f"_{title_suffix.replace(' ', '_')}"
    title = f"{protein} {fm} — Method Comparison{(' ' + title_suffix) if title_suffix else ''}"

    valid_methods = [m for m in methods if m in metrics_by_method and metrics_by_method[m]]

    # ── Figure 1: metrics bar-chart grid ─────────────────────────────────────
    fig_w   = max(20, len(_METRIC_KEYS) * 3.2)
    fig_bar = plt.figure(figsize=(fig_w, 5.0), facecolor="#F8F9FA")
    gs_metrics = fig_bar.add_gridspec(1, len(_METRIC_KEYS), wspace=0.4,
                                      left=0.05, right=0.98,
                                      top=0.82, bottom=0.28)
    metric_axes = [fig_bar.add_subplot(gs_metrics[0, i]) for i in range(len(_METRIC_KEYS))]
    _plot_metrics_grid(metric_axes, valid_methods, metrics_by_method)
    fig_bar.suptitle(title, fontsize=11, fontweight="bold", y=0.97)

    metrics_path = output_dir / f"{base}_metrics.png"
    fig_bar.savefig(metrics_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig_bar)
    print(f"  Saved metrics → {metrics_path}")

    # ── Figure 2: attribution-vs-coverage panels ──────────────────────────────
    cov_results: dict[str, pd.DataFrame] = {}
    for method in methods:
        if method in importance_by_method:
            plm, interp = _decompose_method(method)
            interp_label = _METHOD_LABELS.get(interp, interp).replace("\n", " ")
            if plm == "IGLOO":
                label = interp_label
            else:
                label = f"{interp_label} ({_PLM_LABELS.get(plm, plm)})"
            cov_results[label] = importance_by_method[method]

    coverage_path = output_dir / f"{base}_coverage.png"
    if cov_results:
        plot_attribution_vs_mutation_coverage(
            results=cov_results,
            mut_freq=mut_freq,
            seq_length=seq_length,
            ground_truth=gt,
            uncovered_threshold_pct=uncovered_threshold_pct,
            save_path=str(coverage_path),
            dpi=dpi,
        )
        print(f"  Saved coverage → {coverage_path}")
    else:
        fig_cov, ax_cov = plt.subplots(figsize=(10, 2))
        ax_cov.text(0.5, 0.5, "No coverage data available",
                    ha="center", va="center", fontsize=12, color="gray")
        ax_cov.axis("off")
        fig_cov.savefig(coverage_path, dpi=dpi, bbox_inches="tight", transparent=True)
        plt.close(fig_cov)
        print(f"  Saved coverage (empty) → {coverage_path}")

    return metrics_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Combined attribution-quality + coverage comparison figure "
            "across interpolation and baseline methods."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--results-dirs", nargs="+", type=Path, metavar="DIR",
        help="Explicit list of leaf result directories.",
    )
    src.add_argument(
        "--results-dir-tree", type=Path, metavar="ROOT",
        help="Root directory to scan recursively for result dirs.",
    )

    p.add_argument(
        "--proteins", nargs="+", default=None, metavar="PROTEIN",
        help="Restrict to specific proteins (e.g. GB1). Inferred from path if omitted.",
    )
    p.add_argument(
        "--fitness-models", nargs="+", default=None, metavar="FM",
        help="Restrict to specific fitness models (additive, epistatic, nonlinear).",
    )
    p.add_argument(
        "--datasets-dir", type=Path,
        default=_PROJECT_ROOT / "data" / "synthetic",
        metavar="DIR",
        help="Directory containing {Protein}_{fm}.csv datasets (default: data/synthetic/).",
    )
    p.add_argument(
        "--ground-truth-dir", type=Path,
        default=_PROJECT_ROOT / "data" / "synthetic",
        metavar="DIR",
        help="Directory containing *_ground_truth_weights.json files (default: data/synthetic/).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "results" / "synthetic" / "evaluation" / "method_comparison",
        metavar="DIR",
    )
    p.add_argument(
        "--uncovered-threshold", type=float, default=10.0, metavar="PCT",
        help="Top PCT%% attribution + zero mutations → uncovered hot-spot (default: 10).",
    )
    p.add_argument(
        "--mutation-col", default="mutation_string", metavar="COL",
    )
    p.add_argument(
        "--igloo-only", action="store_true", default=False,
        help="Exclude results from plm_* subdirectories (ankh_base, e1_150m, etc.); "
             "keep only the base ESM-C / IGLOO interpolation results.",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for Friedman / Nemenyi (default: 0.05).")
    p.add_argument("--concordance-threshold", type=float, default=0.6,
                   help="Kendall W cutoff: above → consistent, below → heterogeneous (default: 0.6).")
    return p.parse_args()


def _plot_composite_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
    title: str = "Composite score — methods × scenarios",
) -> None:
    """Save a RdYlGn heatmap: rows = methods/pLMs, columns = scenarios."""
    # matrix: rows = scenarios, cols = methods/pLMs  (like auc_matrix)
    data = matrix.T  # methods × scenarios
    # pretty-print method names using _PLM_LABELS where possible
    def _label(m: str) -> str:
        plm, interp = _decompose_method(m)
        plm_l  = _PLM_LABELS.get(plm, plm)
        interp_l = _METHOD_LABELS.get(interp, interp).replace("\n", " ")
        if plm == "IGLOO" and interp == plm:
            return plm_l  # pure pLM row
        if interp == "":
            return plm_l
        return f"{plm_l} / {interp_l}"

    n_cols = len(data.columns)
    n_rows = len(data.index)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.6), max(4, n_rows * 0.9)))

    im = ax.imshow(data.values.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised composite score")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([s.replace("_", " ") for s in data.columns],
                       fontsize=9, ha="center")
    ax.tick_params(axis="x", which="major", pad=8)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_label(m) for m in data.index], fontsize=10)

    for r in range(n_rows):
        for c in range(n_cols):
            val = data.values[r, c]
            if not np.isnan(float(val)):
                ax.text(c, r, f"{float(val):.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved heatmap → {output_path}")


def _run_plm_stats(
    matrix: pd.DataFrame,
    output_dir: Path,
    dpi: int = 150,
    alpha: float = 0.05,
    concordance_threshold: float = 0.6,
) -> None:
    """Run Friedman + Kendall's W + Nemenyi on a scenario × method composite matrix."""
    if not _STATS_AVAILABLE:
        print("  [stats] igloo.method_selection not available — skipping statistical tests.")
        return
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        print("  [stats] Not enough scenarios/methods for statistical tests — skipping.")
        return

    lines: list[str] = []
    lines.append("Statistical tests — pLM / method composite scores")
    lines.append("=" * 60)

    friedman = run_friedman(matrix, alpha=alpha)
    W        = kendalls_w(matrix)
    lines.append(f"Friedman χ²={friedman['stat']:.3f}  p={friedman['p']:.4f}  "
                 f"({'SIGNIFICANT' if friedman['significant'] else 'not significant'})")
    lines.append(f"Kendall's W = {W:.3f}  (threshold = {concordance_threshold})")
    consistent = W >= concordance_threshold
    lines.append(f"Consistency: {'CONSISTENT' if consistent else 'HETEROGENEOUS'}")
    lines.append("")

    nemenyi = None
    if friedman["significant"]:
        nemenyi = nemenyi_posthoc(matrix)
        if nemenyi is not None:
            lines.append("Nemenyi post-hoc p-values:")
            lines.append(nemenyi.to_string(float_format=lambda v: f"{v:.4f}"))
            lines.append("")
            plot_critical_difference(
                matrix, nemenyi,
                output_dir / "plm_critical_difference.png",
                alpha=alpha, dpi=dpi,
            )

    # Mean composite per method across scenarios
    lines.append("Mean composite per method (across all scenarios):")
    means = matrix.mean(axis=0).sort_values(ascending=False)
    for meth, val in means.items():
        lines.append(f"  {meth:<34} {val:.4f}")

    # ── Paired normal vs deep test ────────────────────────────────────────────
    # Detect paired rows: scenarios ending in "_deep" paired with their base name
    from scipy.stats import wilcoxon, ttest_rel

    deep_rows   = [r for r in matrix.index if str(r).endswith("_deep")]
    normal_rows = [str(r).replace("_deep", "") for r in deep_rows]
    paired      = [(n, d) for n, d in zip(normal_rows, deep_rows)
                   if n in matrix.index and d in matrix.index]

    if paired:
        lines.append("")
        lines.append("Normal vs Deep paired comparison")
        lines.append("-" * 60)
        lines.append(f"Matched pairs: {', '.join(f'{n}↔{d}' for n,d in paired)}")
        lines.append("")

        # Per-method paired test (Wilcoxon where n≥3, else t-test)
        lines.append("Per-method (Wilcoxon signed-rank, or paired t-test if n<4):")
        method_results = []
        for method in matrix.columns:
            normal_scores = [matrix.loc[n, method] for n, d in paired
                             if not np.isnan(matrix.loc[n, method]) and not np.isnan(matrix.loc[d, method])]
            deep_scores   = [matrix.loc[d, method] for n, d in paired
                             if not np.isnan(matrix.loc[n, method]) and not np.isnan(matrix.loc[d, method])]
            if len(normal_scores) < 2:
                lines.append(f"  {method:<34} insufficient data")
                continue
            diff = np.array(normal_scores) - np.array(deep_scores)
            if np.all(diff == 0):
                lines.append(f"  {method:<34} no difference (all ties)")
                continue
            try:
                if len(normal_scores) >= 4:
                    stat, p = wilcoxon(normal_scores, deep_scores, alternative="two-sided")
                    test_name = "Wilcoxon"
                else:
                    stat, p = ttest_rel(normal_scores, deep_scores)
                    test_name = "paired-t"
                direction = "normal>deep" if np.mean(diff) > 0 else "deep>normal"
                sig = "*" if p < alpha else " "
                lines.append(f"  {method:<34} {test_name}  stat={stat:.3f}  p={p:.4f}{sig}  "
                             f"Δmean={np.mean(diff):+.4f} ({direction})")
                method_results.append((method, p, np.mean(diff)))
            except Exception as e:
                lines.append(f"  {method:<34} error: {e}")

        # Pooled test across all methods
        lines.append("")
        lines.append("Pooled test (all methods × pairs concatenated):")
        all_normal, all_deep = [], []
        for method in matrix.columns:
            for n, d in paired:
                nv, dv = matrix.loc[n, method], matrix.loc[d, method]
                if not np.isnan(nv) and not np.isnan(dv):
                    all_normal.append(nv)
                    all_deep.append(dv)
        if len(all_normal) >= 4:
            diff_all = np.array(all_normal) - np.array(all_deep)
            try:
                stat_w, p_w = wilcoxon(all_normal, all_deep, alternative="two-sided")
                stat_t, p_t = ttest_rel(all_normal, all_deep)
                direction = "normal>deep" if np.mean(diff_all) > 0 else "deep>normal"
                lines.append(f"  n_pairs={len(all_normal)}")
                lines.append(f"  Wilcoxon  stat={stat_w:.3f}  p={p_w:.4f}{'*' if p_w < alpha else ''}")
                lines.append(f"  Paired-t  stat={stat_t:.3f}  p={p_t:.4f}{'*' if p_t < alpha else ''}")
                lines.append(f"  Mean Δ (normal−deep) = {np.mean(diff_all):+.4f}  ({direction})")
            except Exception as e:
                lines.append(f"  error: {e}")
        else:
            lines.append("  insufficient pairs for pooled test")
    else:
        lines.append("")
        lines.append("Normal vs Deep paired comparison: no matched pairs found in scenarios.")
        lines.append("  (Expected scenario names like 'GB1_additive' paired with 'GB1_additive_deep')")

    # ── pLM-level comparison ──────────────────────────────────────────────────
    # Detect whether columns have "plm/method" structure; if so, collapse by pLM
    plm_cols: dict[str, list[str]] = {}
    for col in matrix.columns:
        if "/" in str(col):
            plm, _ = str(col).split("/", 1)
        else:
            plm = "ESM-C"   # bare method name → default pLM label
        plm_cols.setdefault(plm, []).append(col)

    if len(plm_cols) >= 2:
        lines.append("")
        lines.append("pLM-level comparison (scores averaged over interpolation methods)")
        lines.append("-" * 60)

        # Build pLM × scenario matrix (average over methods)
        plm_names = sorted(plm_cols.keys())
        plm_matrix = pd.DataFrame(
            {plm: matrix[cols].mean(axis=1) for plm, cols in plm_cols.items()}
        )  # index = scenarios, columns = pLMs

        lines.append("Mean composite per pLM (averaged over methods and scenarios):")
        plm_means = plm_matrix.mean(axis=0).sort_values(ascending=False)
        for plm, val in plm_means.items():
            n_methods = len(plm_cols[plm])
            lines.append(f"  {plm:<20} {val:.4f}  (avg of {n_methods} methods)")

        lines.append("")
        lines.append("Per-scenario means:")
        header = f"  {'scenario':<30}" + "".join(f"  {p:<12}" for p in plm_names)
        lines.append(header)
        for scen in plm_matrix.index:
            row = f"  {str(scen):<30}"
            for p in plm_names:
                v = plm_matrix.loc[scen, p]
                row += f"  {v:<12.4f}"
            lines.append(row)

        lines.append("")
        lines.append("Pairwise pLM tests (paired t-test across scenarios, Bonferroni-corrected):")
        from itertools import combinations
        pairs_plm = list(combinations(plm_names, 2))
        n_comparisons = len(pairs_plm)
        for pA, pB in pairs_plm:
            scores_a = plm_matrix[pA].dropna()
            scores_b = plm_matrix[pB].dropna()
            common_idx = scores_a.index.intersection(scores_b.index)
            a_vals = scores_a.loc[common_idx].values
            b_vals = scores_b.loc[common_idx].values
            if len(a_vals) < 2:
                lines.append(f"  {pA} vs {pB:<20} insufficient data")
                continue
            diff_ab = a_vals - b_vals
            if np.all(diff_ab == 0):
                lines.append(f"  {pA} vs {pB:<20} no difference (all ties)")
                continue
            try:
                if len(a_vals) >= 4:
                    stat, p_raw = wilcoxon(a_vals, b_vals, alternative="two-sided")
                    test_name = "Wilcoxon"
                else:
                    stat, p_raw = ttest_rel(a_vals, b_vals)
                    test_name = "paired-t"
                p_corr = min(p_raw * n_comparisons, 1.0)
                direction = f"{pA}>{pB}" if np.mean(diff_ab) > 0 else f"{pB}>{pA}"
                sig = "*" if p_corr < alpha else " "
                lines.append(
                    f"  {pA} vs {pB:<16} {test_name}  stat={stat:.3f}  "
                    f"p_raw={p_raw:.4f}  p_bonf={p_corr:.4f}{sig}  "
                    f"Δmean={np.mean(diff_ab):+.4f} ({direction})  n={len(a_vals)}"
                )
            except Exception as e:
                lines.append(f"  {pA} vs {pB}: error: {e}")

    txt_path = output_dir / "plm_stats_report.txt"
    txt_path.write_text("\n".join(lines))
    print(f"  Saved stats report → {txt_path}")


def _write_scores_txt(
    protein: str,
    fm: str,
    methods: list[str],
    metrics_by_method: dict[str, dict | None],
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """Write a plain-text table of per-method scores + composite to *_scores.txt."""
    base = f"{protein}_{fm}_method_comparison"
    if title_suffix:
        base += f"_{title_suffix.replace(' ', '_')}"
    out_path = output_dir / f"{base}_scores.txt"

    lines: list[str] = []
    header_label = f"{protein} {fm}" + (f" [{title_suffix}]" if title_suffix else "")
    lines.append(f"Composite & metric scores — {header_label}")
    lines.append("=" * 80)

    col_w = 12
    short_keys = ["ndcg", "ap_graded", "auroc", "pv_ndcg", "sign", "cat_sep", "mut_ndcg"]
    header = f"{'Method':<34}" + "".join(f"{k:>{col_w}}" for k in short_keys) + f"{'Composite':>{col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    valid = [(m, metrics_by_method[m]) for m in methods
             if m in metrics_by_method and metrics_by_method[m]]

    # Build raw matrix: shape (n_methods, n_metrics)
    raw = np.array([[m.get(k, float("nan")) for k in _METRIC_KEYS] for _, m in valid])

    # Min-max normalise each metric column across methods (same as add_composite_score)
    normed = raw.copy()
    for ci in range(raw.shape[1]):
        col = raw[:, ci]
        vmin, vmax = np.nanmin(col), np.nanmax(col)
        if vmax > vmin:
            normed[:, ci] = (col - vmin) / (vmax - vmin)
        else:
            normed[:, ci] = np.where(np.isfinite(col), 0.5, float("nan"))

    composites = np.nanmean(normed, axis=1)

    for i, (method, m) in enumerate(valid):
        vals = raw[i]
        composite = composites[i]
        row = f"{method:<34}" + "".join(f"{v:>{col_w}.4f}" for v in vals) + f"{composite:>{col_w}.4f}"
        lines.append(row)

    lines.append("")
    lines.append("  Note: Composite = mean of min-max-normalised metrics (per metric, across methods shown).")

    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"  Saved scores  → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Accumulates composite scores for the cross-group heatmap + stats
    # Each record: {"method": str, "scenario": str, "composite": float}
    all_score_records: list[dict] = []

    # ── Collect directories ───────────────────────────────────────────────────
    if args.results_dirs:
        dirs = [Path(d) for d in args.results_dirs]
        if args.igloo_only:
            dirs = [d for d in dirs if not any(part.startswith("plm_") for part in d.parts)]
    else:
        print(f"Scanning {args.results_dir_tree} …")
        dirs = _discover_leaf_dirs(args.results_dir_tree, igloo_only=args.igloo_only)
        print(f"  Found {len(dirs)} directories.")

    if not dirs:
        print("No result directories found.")
        sys.exit(0)

    # ── Group by protein × fitness-model ─────────────────────────────────────
    groups: dict[tuple[str, str], list[Path]] = {}
    for d in dirs:
        prot, fm = _infer_protein_and_fm(d)
        if prot is None:
            print(f"  [warn] Cannot infer protein/FM from {d} — skipping.")
            continue
        if args.proteins and prot not in args.proteins:
            continue
        if args.fitness_models and fm not in args.fitness_models:
            continue
        groups.setdefault((prot, fm), []).append(d)

    if not groups:
        print("No matching groups found after filters.")
        sys.exit(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each group ────────────────────────────────────────────────────
    for (protein, fm), group_dirs in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Group: {protein} / {fm}  ({len(group_dirs)} dirs)")

        # Load dataset CSV
        dataset_csv = args.datasets_dir / f"{protein}_{fm}.csv"
        if not dataset_csv.exists():
            print(f"  [warn] Dataset not found: {dataset_csv} — skipping coverage panel.")
            mut_freq  = pd.Series(dtype=int)
            seq_len   = 0
        else:
            dataset_df = pd.read_csv(dataset_csv)
            mut_freq   = _mutation_frequency(dataset_df, mutation_col=args.mutation_col)
            seq_len    = _infer_seq_length(dataset_df, None)
            print(f"  Dataset: {len(dataset_df):,} variants, seq_length={seq_len}")

        # Load ground-truth JSON — try {protein}_{fm} first, then protein-only
        gt_path = args.ground_truth_dir / f"{protein}_{fm}_ground_truth_weights.json"
        if not gt_path.exists():
            gt_path = args.ground_truth_dir / f"{protein}_ground_truth_weights.json"
        gt: dict | None = None
        if gt_path.exists():
            with open(gt_path) as fh:
                gt = json.load(fh)
            if gt and "sequence_length" in gt:
                seq_len = int(gt["sequence_length"])
            print(f"  Ground truth: {gt_path.name}")
        else:
            print(f"  [info] No ground-truth JSON found for {protein}/{fm}")

        # Infer method name from last path component, prefixed with pLM name if present
        def _method(d: Path) -> str:
            for part in d.parts:
                if part.startswith("plm_"):
                    plm_tag = part[4:]          # strip "plm_" prefix
                    return f"{plm_tag}/{d.name}"
            return d.name

        # Build metrics + importance lookup keyed by method name
        metrics_by_method:    dict[str, dict | None]  = {}
        importance_by_method: dict[str, pd.DataFrame] = {}

        for d in group_dirs:
            method = _method(d)
            print(f"  [{method}] computing metrics …", end=" ", flush=True)

            # Quality metrics
            m = compute_metrics_for_dir(
                d, protein,
                fitness_model=fm,
                split_filter=None,
            )
            metrics_by_method[method] = m
            if m:
                print(
                    f"NDCG={m.get('ndcg', float('nan')):.3f}  "
                    f"AP@R={m.get('ap_graded', float('nan')):.3f}  "
                    f"AUROC={m.get('auroc', float('nan')):.3f}  "
                    f"pvNDCG={m.get('pv_ndcg_mean', float('nan')):.3f}  "
                    f"sign={m.get('sign_acc', float('nan')):.3f}  "
                    f"catSep={m.get('cat_sep', float('nan')):.3f}  "
                    f"mutNDCG={m.get('mutation_ndcg', float('nan')):.3f}"
                )
            else:
                print("no data")

            # Importance CSV for coverage panel
            csv = _find_importance_csv(d)
            if csv is not None:
                importance_by_method[method] = pd.read_csv(csv)

        # Determine method order for this group
        methods = [m for m in _METHOD_ORDER if m in metrics_by_method or m in importance_by_method]
        # Append any extra methods not in _METHOD_ORDER
        for m in sorted(metrics_by_method):
            if m not in methods:
                methods.append(m)

        # Build title suffix from noise level if detectable
        noise_parts = [p for p in group_dirs[0].parts if p.startswith("noise_")]
        title_suffix = noise_parts[0] if noise_parts else ""

        _make_comparison_figure(
            protein=protein,
            fm=fm,
            methods=methods,
            metrics_by_method=metrics_by_method,
            importance_by_method=importance_by_method,
            mut_freq=mut_freq,
            seq_length=seq_len,
            gt=gt,
            uncovered_threshold_pct=args.uncovered_threshold,
            output_dir=args.output_dir,
            dpi=args.dpi,
            title_suffix=title_suffix,
        )
        _write_scores_txt(
            protein=protein,
            fm=fm,
            methods=methods,
            metrics_by_method=metrics_by_method,
            output_dir=args.output_dir,
            title_suffix=title_suffix,
        )

        # ── Accumulate normalised composites for summary heatmap + stats ─────
        valid_for_heatmap = [
            (m, metrics_by_method[m]) for m in methods
            if m in metrics_by_method and metrics_by_method[m]
        ]
        if valid_for_heatmap:
            raw_h = np.array([[mv.get(k, float("nan")) for k in _METRIC_KEYS]
                              for _, mv in valid_for_heatmap])
            normed_h = raw_h.copy()
            for ci in range(raw_h.shape[1]):
                col = raw_h[:, ci]
                vmin, vmax = np.nanmin(col), np.nanmax(col)
                if vmax > vmin:
                    normed_h[:, ci] = (col - vmin) / (vmax - vmin)
                else:
                    normed_h[:, ci] = np.where(np.isfinite(col), 0.5, float("nan"))
            comps_h = np.nanmean(normed_h, axis=1)
            scenario_label = f"{protein}_{fm}"
            for i, (method, _) in enumerate(valid_for_heatmap):
                all_score_records.append({
                    "method": method,
                    "scenario": scenario_label,
                    "composite": float(comps_h[i]),
                })

    # ── Summary heatmap + statistical tests across all groups ─────────────────
    if all_score_records:
        rec_df  = pd.DataFrame(all_score_records)
        # Pivot: rows = scenarios, columns = methods
        pivot = rec_df.pivot_table(
            index="scenario", columns="method", values="composite", aggfunc="mean"
        )
        _plot_composite_heatmap(
            pivot,
            args.output_dir / "composite_score_heatmap.png",
            dpi=args.dpi,
            title="Composite score — methods × scenarios",
        )
        _run_plm_stats(
            pivot,
            args.output_dir,
            dpi=args.dpi,
            alpha=args.alpha,
            concordance_threshold=args.concordance_threshold,
        )

    print(f"\nDone. Output → {args.output_dir}")


if __name__ == "__main__":
    main()
