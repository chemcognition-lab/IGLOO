#!/usr/bin/env python3
"""
Robust Synthetic IG Validation Script
======================================
A bias-corrected parallel to evaluate_synthetic_ig.py with four fixes that
prevent survivorship bias and artificially inflated metrics at high noise levels.

Fixes applied
-------------
1. **Penalize near-zero sequences** – ``compute_completeness_stats()`` no longer
   silently drops sequences with |F(x)−F(x')| < threshold.  They are instead
   treated as worst-case completeness failures (error = 1.0), so completeness
   degrades visibly at noise=1 rather than appearing artificially clean.

2. **Filter collapsed sequences from position aggregation** –
   ``_load_per_variant_gradients_robust()`` excludes sequences with
   |effect_vs_baseline| < min_output_diff from the position-level mean, and
   returns an accompanying per-position count array.  When the model has
   collapsed entirely, the function returns (None, None) — an honest failure
   signal rather than a silent average of near-zero gradients.

3. **Model-collapse diagnostic metric** – ``compute_model_collapse_stats()``
   reports ``collapse_fraction`` (fraction of non-WT sequences with near-zero
   output difference).  This is included in the summary CSV and can be used to
   shade unreliable regions of noise-vs-metric plots.

4. **Reliability mask in compute_metrics()** – ``compute_metrics_robust()``
   accepts an optional ``position_counts`` array and silently drops positions
   covered by fewer than ``min_count`` non-collapsed sequences before computing
   Spearman ρ, AUROC, etc.  If fewer than 10 reliable positions remain, all
   metrics are returned as NaN rather than computed on sparse data.

Usage
-----
    source IGLOO_env/bin/activate

    # Drop-in replacement for evaluate_synthetic_ig.py:
    python scripts/evaluation/evaluate_synthetic_ig_robust.py --use-per-variant

    # All the same flags as the original:
    python scripts/evaluation/evaluate_synthetic_ig_robust.py \\
        --use-per-variant \\
        --proteins GB1 \\
        --fitness-models additive epistatic nonlinear \\
        --output-dir results/synthetic/evaluation_robust

    # Output: results/synthetic/evaluation_robust/
    #   synthetic_ig_validation_summary.csv  – same schema as original, plus:
    #     collapse_fraction      – fraction of non-WT seqs with near-zero |ΔF|
    #     n_total_sequences      – total non-WT sequences seen
    #   (all other plots are identical to the original)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Import shared infrastructure from the sibling script
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# We also need the project root on the path for igloo imports
_PROJECT_ROOT = _SCRIPTS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import evaluate_synthetic_ig as _base

# Re-export everything we don't change so the rest of the code can use it
from evaluate_synthetic_ig import (
    # constants
    PROJECT_ROOT,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_RESULTS_DIR,
    PROTEINS,
    BACKGROUND_IMPORTANCE,
    CLIFF_CRITICAL_CATEGORIES,
    _NEAR_ZERO_THRESHOLD,
    # ground-truth helpers
    load_ground_truth,
    # loading helpers
    _load_position_importance,
    _load_detailed_ig,
    _is_wt_sequence_name,
    _extract_split_tag,
    _get_wt_sequence,
    _mutations_from_ig,
    _load_data_csv,
    # unchanged metric functions
    compute_fitness_based_metrics,
    compute_sign_correctness,
    compute_normalized_position_scores,
    compute_precision_at_k,
    compute_linear_proportionality,
    compute_epistatic_recovery,
    compute_cliff_analysis,
    compute_per_variant_gt_correlation,
    compute_per_variant_fitness_contribution_correlation,
    # discover + plot helpers
    discover_synthetic_results,
)

# Plotting imports (same as original)
from igloo.plot import (
    plot_sign_accuracy_bars,
    plot_cliff_damage,
    plot_metric_comparison,
    plot_fitness_metric_comparison,
    plot_completeness_summary,
    plot_precision_at_k as _plot_precision_at_k,
    plot_noise_vs_interp,
    plot_noise_vs_interp_boxplots,
    plot_per_position_comparison,
)


# ---------------------------------------------------------------------------
# FIX 1 – Penalize near-zero sequences instead of excluding them
# ---------------------------------------------------------------------------

def compute_completeness_stats(ig_df: pd.DataFrame) -> dict:
    """Check the IG completeness axiom: Σ_i IG_i ≈ F(x) − F(x').

    **Robust change vs original**: sequences with |outputDiff| below
    _NEAR_ZERO_THRESHOLD are no longer silently dropped.  They are treated as
    worst-case completeness failures (relative error = 1.0) so that at noise=1,
    where many sequences collapse, completeness degrades rather than appearing
    artificially clean.

    Returns
    -------
    dict with:
      completeness_mean_rel_error   – mean ε over ALL non-WT sequences
      completeness_median_rel_error
      completeness_frac_pass_1pct   – fraction with ε < 0.01
      completeness_frac_pass_5pct   – fraction with ε < 0.05
      completeness_frac_pass_10pct  – fraction with ε < 0.10
      n_completeness_sequences      – total non-WT sequences evaluated
      n_near_zero_sequences         – sequences penalised (|outputDiff| < threshold)
    """
    _nan = {
        "completeness_mean_rel_error":   float("nan"),
        "completeness_median_rel_error": float("nan"),
        "completeness_frac_pass_1pct":   float("nan"),
        "completeness_frac_pass_5pct":   float("nan"),
        "completeness_frac_pass_10pct":  float("nan"),
        "n_completeness_sequences":      0,
        "n_near_zero_sequences":         0,
    }

    need_cols = {"integrated_gradient", "effect_vs_baseline"}
    if not need_cols.issubset(ig_df.columns):
        return _nan

    use_unclipped = "effect_vs_baseline_unclipped" in ig_df.columns
    diff_col = "effect_vs_baseline_unclipped" if use_unclipped else "effect_vs_baseline"

    errors: list[float] = []
    n_near_zero = 0

    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        output_diff = float(seq_df[diff_col].iloc[0])
        if abs(output_diff) < _NEAR_ZERO_THRESHOLD:
            n_near_zero += 1
            errors.append(1.0)  # ← FIX: treat as worst-case error, not excluded
            continue
        sum_ig = float(seq_df["integrated_gradient"].sum())
        rel_err = abs(sum_ig - output_diff) / abs(output_diff)
        errors.append(rel_err)

    n = len(errors)
    if n == 0:
        return {**_nan, "n_near_zero_sequences": n_near_zero}

    err_arr = np.array(errors)
    return {
        "completeness_mean_rel_error":   float(err_arr.mean()),
        "completeness_median_rel_error": float(np.median(err_arr)),
        "completeness_frac_pass_1pct":   float((err_arr < 0.01).mean()),
        "completeness_frac_pass_5pct":   float((err_arr < 0.05).mean()),
        "completeness_frac_pass_10pct":  float((err_arr < 0.10).mean()),
        "n_completeness_sequences":      n,
        "n_near_zero_sequences":         n_near_zero,
    }


# ---------------------------------------------------------------------------
# FIX 2 – Filter collapsed sequences from position-level aggregation,
#          return per-position count array alongside mean scores
# ---------------------------------------------------------------------------

def _load_per_variant_gradients_robust(
    results_dir: Path,
    dataset_name: str | None = None,
    split_filter: str | None = None,
    min_output_diff: float = _NEAR_ZERO_THRESHOLD,
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Load per-variant IG CSV, filter collapsed sequences, and aggregate.

    **Robust changes vs original**:
    - Sequences with |effect_vs_baseline| < min_output_diff are excluded from
      the mean so that near-zero gradients from collapsed sequences do not
      dilute position-level signals.
    - Returns a second element: a (L_max,) int32 array of per-position counts
      (how many non-collapsed sequences contributed to each position's mean).
      This is used by compute_metrics_robust() to apply a reliability mask.
    - If every sequence is collapsed, returns (None, None) explicitly.

    Parameters
    ----------
    results_dir     : directory containing *_integrated_gradients_results.csv
    dataset_name    : optional prefix for the CSV filename
    split_filter    : 'trainval', 'test', or None
    min_output_diff : |ΔF| threshold; sequences below this are dropped

    Returns
    -------
    (agg_df, count_arr) where agg_df has columns [position, mean_abs_gradient]
    and count_arr is a 1-D int array indexed by (position - 1).
    Both are None if no data found or all sequences collapsed.
    """
    candidates = list(results_dir.glob("*_integrated_gradients_results.csv"))
    if not candidates and dataset_name:
        candidates = [results_dir / f"{dataset_name}_integrated_gradients_results.csv"]
    if not candidates:
        return None, None
    path = candidates[0]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Apply split filter
    if split_filter and split_filter in ("trainval", "test") and "sequence_name" in df.columns:
        mask = df["sequence_name"].apply(lambda n: _extract_split_tag(n) == split_filter)
        df = df[mask]
        if df.empty:
            return None, None

    # ── FIX: drop collapsed sequences (|ΔF| too small) ──────────────────────
    if "effect_vs_baseline" in df.columns:
        # Compute per-sequence |ΔF| (constant within a sequence)
        seq_diff = (
            df.groupby("sequence_name")["effect_vs_baseline"]
            .first()
            .abs()
        )
        valid_seqs = seq_diff[seq_diff >= min_output_diff].index
        df = df[df["sequence_name"].isin(valid_seqs)]
        if df.empty:
            return None, None  # all sequences collapsed — honest failure signal

    # Aggregate: mean abs_gradient per position, plus per-position counts
    agg = (
        df.groupby("position")["abs_gradient"]
        .agg(mean_abs_gradient="mean", count="count")
        .reset_index()
    )

    # Also aggregate mean signed (non-abs) gradient per position
    if "integrated_gradient" in df.columns:
        agg_signed = (
            df.groupby("position")["integrated_gradient"]
            .mean()
            .reset_index()
            .rename(columns={"integrated_gradient": "mean_signed_gradient"})
        )
        agg = agg.merge(agg_signed, on="position", how="left")
    else:
        agg["mean_signed_gradient"] = agg["mean_abs_gradient"]

    # Build count array indexed by (position - 1), length = max_position
    max_pos = int(agg["position"].max())
    count_arr = np.zeros(max_pos, dtype=np.int32)
    for _, row in agg.iterrows():
        pos_0 = int(row["position"]) - 1
        if 0 <= pos_0 < max_pos:
            count_arr[pos_0] = int(row["count"])

    pos_df = agg[["position", "mean_abs_gradient", "mean_signed_gradient"]].copy()
    return pos_df, count_arr


# ---------------------------------------------------------------------------
# FIX 3 – Model-collapse diagnostic metric
# ---------------------------------------------------------------------------

def compute_model_collapse_stats(ig_df: pd.DataFrame) -> dict:
    """Fraction of non-WT sequences where |F(x) − F(x')| < threshold.

    High ``collapse_fraction`` (> ~0.3) indicates that the model has collapsed
    and all attribution metrics computed on those sequences are unreliable.

    Returns
    -------
    dict with:
      collapse_fraction   – n_collapsed / n_total (0–1)
      n_total_sequences   – total non-WT sequences
      n_collapsed_seqs    – sequences with |ΔF| < threshold
    """
    _nan = {
        "collapse_fraction":  float("nan"),
        "n_total_sequences":  0,
        "n_collapsed_seqs":   0,
    }

    if "effect_vs_baseline" not in ig_df.columns:
        return _nan

    n_total = 0
    n_collapsed = 0

    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        n_total += 1
        if abs(float(seq_df["effect_vs_baseline"].iloc[0])) < _NEAR_ZERO_THRESHOLD:
            n_collapsed += 1

    return {
        "collapse_fraction":  n_collapsed / n_total if n_total > 0 else float("nan"),
        "n_total_sequences":  n_total,
        "n_collapsed_seqs":   n_collapsed,
    }


# ---------------------------------------------------------------------------
# FIX 4 – Reliability mask in compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics_robust(
    ig_scores: np.ndarray,
    gt_importance: np.ndarray,
    annotated_mask: np.ndarray,
    position_counts: np.ndarray | None = None,
    min_count: int = 5,
) -> dict:
    """Compute all validation metrics, optionally masking low-coverage positions.

    **Robust changes vs original**:
    - Accepts an optional ``position_counts`` array (number of non-collapsed
      sequences that contributed to each position's mean |IG|).
    - Positions covered by fewer than ``min_count`` sequences are excluded
      before computing Spearman ρ, AUROC, etc.
    - If fewer than 10 reliable positions remain after masking, all metrics
      are returned as NaN rather than being computed on sparse data.

    Parameters
    ----------
    ig_scores       : (L,) array of |IG| values per position (0-indexed).
    gt_importance   : (L,) ground-truth importance vector.
    annotated_mask  : (L,) bool array; True = functionally annotated.
    position_counts : (L,) int array or None.  When provided, positions with
                      count < min_count are excluded.
    min_count       : minimum number of non-collapsed sequences required for a
                      position to be included in rank-correlation metrics.

    Returns
    -------
    dict with keys: spearman_r, spearman_p, kendall_tau, kendall_p,
                    auroc, auprc, n_positions, n_annotated,
                    precision_at_k, precision_at_k_random,
                    n_reliable_positions  (new key).
    """
    assert len(ig_scores) == len(gt_importance) == len(annotated_mask), (
        f"Length mismatch: ig={len(ig_scores)}, gt={len(gt_importance)}, "
        f"mask={len(annotated_mask)}"
    )

    _nan_result = {
        "spearman_r":             float("nan"),
        "spearman_p":             float("nan"),
        "kendall_tau":            float("nan"),
        "kendall_p":              float("nan"),
        "auroc":                  float("nan"),
        "auprc":                  float("nan"),
        "n_positions":            int(len(ig_scores)),
        "n_annotated":            int(annotated_mask.sum()),
        "precision_at_k":         float("nan"),
        "precision_at_k_random":  float("nan"),
        "n_reliable_positions":   0,
    }

    # ── FIX: apply reliability mask ──────────────────────────────────────────
    if position_counts is not None:
        # Align count array length with ig_scores if needed
        L = len(ig_scores)
        if len(position_counts) < L:
            padded = np.zeros(L, dtype=np.int32)
            padded[:len(position_counts)] = position_counts
            position_counts = padded
        else:
            position_counts = position_counts[:L]

        reliable = position_counts >= min_count
        n_reliable = int(reliable.sum())

        if n_reliable < 10:
            _nan_result["n_reliable_positions"] = n_reliable
            return _nan_result

        ig_scores      = ig_scores[reliable]
        gt_importance  = gt_importance[reliable]
        annotated_mask = annotated_mask[reliable]
    else:
        n_reliable = int(len(ig_scores))

    # Rank correlations — over reliable positions
    sp_r, sp_p   = spearmanr(ig_scores, gt_importance)
    kt_tau, kt_p = kendalltau(ig_scores, gt_importance)

    # Classification metrics — annotated vs unannotated
    y_true  = annotated_mask.astype(int)
    y_score = ig_scores

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    # Precision@K: K = n_annotated (among reliable positions)
    K = int(y_true.sum())
    if K > 0 and K < len(y_score):
        top_k_idx = np.argsort(-y_score)[:K]
        precision_at_k = float(y_true[top_k_idx].mean())
        precision_at_k_random = K / len(y_score)
    else:
        precision_at_k = float("nan")
        precision_at_k_random = float("nan")

    return {
        "spearman_r":             float(sp_r),
        "spearman_p":             float(sp_p),
        "kendall_tau":            float(kt_tau),
        "kendall_p":              float(kt_p),
        "auroc":                  float(auroc),
        "auprc":                  float(auprc),
        "n_positions":            int(len(ig_scores)),
        "n_annotated":            int(y_true.sum()),
        "precision_at_k":         precision_at_k,
        "precision_at_k_random":  precision_at_k_random,
        "n_reliable_positions":   n_reliable,
    }


# ---------------------------------------------------------------------------
# FIX 5 – Within-category discrimination (immune to annotation-clustering bias)
# ---------------------------------------------------------------------------

def compute_within_category_discrimination(
    ig_vec: np.ndarray,
    gt: dict,
) -> dict:
    """Within-category Spearman ρ between |IG| and ground-truth importance.

    Global Spearman ρ can be artificially elevated when annotation categories
    form spatially clustered plateaus (e.g. GB1 binding-site positions 9-16,
    26-37, 39-42).  A collapsed model that uniformly attends to that region
    overlaps heavily with the annotated set without actually discriminating
    *within* the category.

    This metric asks only: can the model rank positions within each category by
    their individual importance weight?  A collapsed model outputs
    near-identical attributions for all positions in a category and therefore
    scores near-zero (or exactly 0.0 when all values are identical), even if
    its global ρ remains elevated.

    Returns
    -------
    dict with one key per category: ``within_{category}_spearman``.
    NaN when fewer than 3 positions are in the category; 0.0 when the model
    has collapsed (all |IG| values identical within the category).
    """
    annotated_positions: dict = gt.get("annotated_positions", {})
    importance_vec: np.ndarray = gt["importance_vector"]

    results: dict = {}
    for category, positions in annotated_positions.items():
        key = f"within_{category}_spearman"
        if len(positions) < 3:
            results[key] = float("nan")
            continue

        pos_arr = np.array(positions, dtype=int)
        # Guard against out-of-bounds positions
        valid = pos_arr[pos_arr < len(ig_vec)]
        if len(valid) < 3:
            results[key] = float("nan")
            continue

        ig_sub = ig_vec[valid]
        gt_sub = importance_vec[valid]

        # Collapsed model: all attributions identical → can't discriminate
        if np.all(ig_sub == ig_sub[0]):
            results[key] = 0.0
            continue

        # Skip if ground-truth is constant within category (no signal to rank)
        if np.all(gt_sub == gt_sub[0]):
            results[key] = float("nan")
            continue

        sp_r, _ = spearmanr(ig_sub, gt_sub)
        results[key] = float(sp_r)

    return results


# ---------------------------------------------------------------------------
# Robust evaluate_one – wires the four fixes together
# ---------------------------------------------------------------------------

def evaluate_one(
    results_dir: Path,
    use_per_variant: bool = False,
    split_filter: str | None = None,
) -> dict | None:
    """Evaluate a single results directory (robust version).

    Identical to the original ``evaluate_one`` except:
    - Uses ``_load_per_variant_gradients_robust()`` (fix 2).
    - Passes ``position_counts`` to ``compute_metrics_robust()`` (fix 4).
    - Calls ``compute_completeness_stats()`` from this module (fix 1).
    - Calls ``compute_model_collapse_stats()`` (fix 3) and includes
      ``collapse_fraction`` in the returned dict.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"  [skip] Not a directory: {results_dir}")
        return None

    # ── Infer protein ────────────────────────────────────────────────────────
    full_path_str = str(results_dir)
    protein = None
    for part in reversed(results_dir.parts):
        for p in sorted(PROTEINS, key=len, reverse=True):
            if p.lower() in part.lower():
                protein = p
                break
        if protein:
            break
    if protein is None:
        print(f"  [skip] Cannot identify protein from path: {results_dir}")
        return None

    # ── Parse metadata ───────────────────────────────────────────────────────
    dirname       = results_dir.name
    noise_dirname = results_dir.parent.name
    model_dirname = results_dir.parent.parent.name

    fitness_model = next(
        (fm for fm in ["additive", "epistatic", "nonlinear"]
         if fm in model_dirname.lower() or fm in dirname.lower()),
        "unknown",
    )
    ig_method = next(
        (m for m in ["embedding", "token"] if m in full_path_str.lower()),
        "unknown",
    )
    attribution = next(
        (a for a in ["signed", "magnitude"] if a in full_path_str.lower()),
        "unknown",
    )
    _interp_map = {
        "nearest_neighbor":  "nearest_neighbor",
        "slerp":             "slerp",
        "linear":            "linear",
        "guided_ig":         "guided_ig",
        "gradient_x_input":  "gradient_x_input",
        "input_difference":  "input_difference",
        "attention_weights": "attention_weights",
    }
    interp = _interp_map.get(dirname.lower(), None)
    if interp is None:
        m = re.search(
            r"(nearest_neighbor|slerp|linear|guided_ig|linearinterpolation)",
            full_path_str, re.IGNORECASE,
        )
        interp = m.group(1).lower() if m else "unknown"

    noise_match = re.search(r"noise[_\-]([0-9]*\.?[0-9]+)", noise_dirname, re.IGNORECASE)
    noise_level: float | None = float(noise_match.group(1)) if noise_match else None

    # ── Load ground truth ────────────────────────────────────────────────────
    try:
        gt = load_ground_truth(protein)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    L    = gt["sequence_length"]
    gt_vec = gt["importance_vector"]
    gt_mask = gt["annotated_mask"]

    # ── Load IG position scores (robust) ─────────────────────────────────────
    position_counts: np.ndarray | None = None

    if use_per_variant:
        pos_df, position_counts = _load_per_variant_gradients_robust(
            results_dir, split_filter=split_filter
        )
    else:
        pos_df = _load_position_importance(results_dir)
        # No count array available for pre-aggregated summary

    if pos_df is None or pos_df.empty:
        print(f"  [skip] No position data found in {results_dir}")
        return None

    # Align CSV (1-indexed) to ground-truth (0-indexed, length L)
    ig_vec        = np.zeros(L, dtype=float)
    ig_vec_signed = np.zeros(L, dtype=float)
    _has_signed   = "mean_signed_gradient" in pos_df.columns
    for _, row in pos_df.iterrows():
        pos_0 = int(row["position"]) - 1
        if 0 <= pos_0 < L:
            ig_vec[pos_0]        = float(row["mean_abs_gradient"])
            ig_vec_signed[pos_0] = float(
                row["mean_signed_gradient"] if _has_signed else row["mean_abs_gradient"]
            )

    # ── FIX 4: compute metrics with reliability mask ─────────────────────────
    # Trim / pad count array to match ground-truth length L
    counts_for_metrics: np.ndarray | None = None
    if position_counts is not None:
        if len(position_counts) < L:
            padded = np.zeros(L, dtype=np.int32)
            padded[:len(position_counts)] = position_counts
            counts_for_metrics = padded
        else:
            counts_for_metrics = position_counts[:L]

    metrics = compute_metrics_robust(ig_vec, gt_vec, gt_mask,
                                     position_counts=counts_for_metrics)
    weighted_acc_metrics = compute_precision_at_k(ig_vec, gt_mask)

    # ── Detailed per-sequence IG for advanced metrics ────────────────────────
    ig_df_full = _load_detailed_ig(results_dir)
    wt_seq = _get_wt_sequence(ig_df_full) if ig_df_full is not None else None

    ig_df = ig_df_full
    if ig_df is not None and split_filter and split_filter in ("trainval", "test"):
        split_mask = ig_df["sequence_name"].apply(
            lambda n: _extract_split_tag(n) == split_filter
        )
        ig_df = ig_df[split_mask]
        if ig_df.empty:
            ig_df = None

    # ── FIX 3: model-collapse diagnostic ────────────────────────────────────
    collapse_metrics: dict = {}
    if ig_df is not None:
        collapse_metrics = compute_model_collapse_stats(ig_df)

    # ── FIX 1: penalised completeness stats ──────────────────────────────────
    completeness_metrics: dict = {}
    if ig_df is not None:
        completeness_metrics = compute_completeness_stats(ig_df)   # ← patched version

    # ── Remaining metrics (unchanged from original) ──────────────────────────
    sign_metrics: dict = {}
    if ig_df is not None and wt_seq is not None:
        sign_metrics = compute_sign_correctness(ig_df, gt, wt_seq)

    prop_metrics: dict = {}
    if ig_df is not None and wt_seq is not None:
        prop_metrics = compute_linear_proportionality(ig_df, gt, wt_seq)

    epi_metrics: dict = {}
    if ig_df is not None:
        epi_metrics = compute_epistatic_recovery(ig_df, gt, protein)

    cliff_result: dict = {"cliff_rows": [], "cliff_threshold": 1.5}
    if ig_df is not None and wt_seq is not None:
        data_df = _load_data_csv(protein, fitness_model)
        cliff_result = compute_cliff_analysis(ig_df, gt, protein, data_df)

    fitness_metrics = compute_fitness_based_metrics(ig_vec, protein, fitness_model, wt_seq)

    norm_metrics: dict = {}
    if ig_df is not None:
        norm_ig_vec = compute_normalized_position_scores(ig_df, L)
        if norm_ig_vec is not None:
            raw = compute_metrics_robust(norm_ig_vec, gt_vec, gt_mask)
            norm_metrics = {
                f"norm_{k}": v for k, v in raw.items()
                if k in ("spearman_r", "kendall_tau", "auroc", "auprc")
            }

    pv_gt_metrics: dict = {}
    if ig_df is not None:
        pv_gt_metrics = compute_per_variant_gt_correlation(ig_df, gt)

    pvfc_metrics: dict = {}
    if ig_df is not None and wt_seq is not None:
        pvfc_metrics = compute_per_variant_fitness_contribution_correlation(
            ig_df, gt, fitness_model, wt_seq, protein_name=protein
        )

    # ── FIX 5: within-category discrimination ────────────────────────────────
    within_cat_metrics: dict = compute_within_category_discrimination(ig_vec, gt)

    # ── Signed IG duplicates (mean raw integrated gradient, not abs) ──────────
    metrics_signed_raw = compute_metrics_robust(
        ig_vec_signed, gt_vec, gt_mask, position_counts=counts_for_metrics
    )
    metrics_signed: dict = {
        f"{k}_signed": v for k, v in metrics_signed_raw.items()
        if k not in ("n_positions", "n_annotated")
    }
    within_cat_signed: dict = {
        f"{k}_signed": v
        for k, v in compute_within_category_discrimination(ig_vec_signed, gt).items()
    }

    return {
        "results_dir":   str(results_dir),
        "dirname":       dirname,
        "protein":       protein,
        "fitness_model": fitness_model,
        "ig_method":     ig_method,
        "attribution":   attribution,
        "interp":        interp,
        "noise_level":   noise_level,
        "split":         split_filter if split_filter else "all",
        **metrics,
        **weighted_acc_metrics,
        **norm_metrics,
        **pv_gt_metrics,
        **pvfc_metrics,
        **completeness_metrics,
        **collapse_metrics,        # ← fix 3: collapse_fraction, n_total_sequences
        **within_cat_metrics,       # ← fix 5: within_{category}_spearman
        **metrics_signed,           # ← signed IG duplicates of core position metrics
        **within_cat_signed,        # ← signed IG duplicates of within-category metrics
        **sign_metrics,
        **prop_metrics,
        **epi_metrics,
        **fitness_metrics,
        "_cliff_rows":      cliff_result["cliff_rows"],
        "_cliff_threshold": cliff_result["cliff_threshold"],
    }


# ---------------------------------------------------------------------------
# Main  (identical to original except calls robust evaluate_one)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Robust variant of evaluate_synthetic_ig.py — applies bias-correction "
            "fixes for near-zero / collapsed sequences."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dirs",
        nargs="*",
        help="Results directories to evaluate. If omitted, all synthetic results are "
             "auto-discovered under results/synthetic/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SYNTHETIC_RESULTS_DIR / "evaluation_robust"),
        help="Where to write summary CSV and plots (default: results/synthetic/evaluation_robust).",
    )
    parser.add_argument(
        "--use-per-variant",
        action="store_true",
        help="Use per-variant integrated_gradients_results.csv (recommended).",
    )
    parser.add_argument(
        "--per-position-plots",
        action="store_true",
        help="Also save a per-position |IG| vs ground-truth bar chart for each run.",
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        default=None,
        choices=["GB1", "T4Lysozyme", "p53"],
        help="Restrict auto-discovery to specific proteins (default: all).",
    )
    parser.add_argument(
        "--fitness-models",
        nargs="+",
        default=None,
        choices=["additive", "epistatic", "nonlinear"],
        help="Restrict auto-discovery to specific fitness models (default: all).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum non-collapsed sequences per position for reliability mask (default: 5).",
    )
    parser.add_argument(
        "--min-output-diff",
        type=float,
        default=_NEAR_ZERO_THRESHOLD,
        help=f"Threshold for considering a sequence collapsed (default: {_NEAR_ZERO_THRESHOLD}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover / validate results directories ──────────────────────────────
    if args.results_dirs:
        dirs = [Path(d) for d in args.results_dirs]
    else:
        print(f"Scanning for synthetic results under {SYNTHETIC_RESULTS_DIR} ...")
        all_dirs = discover_synthetic_results()

        def _matches(d: Path) -> bool:
            parts = d.relative_to(SYNTHETIC_RESULTS_DIR).parts
            top = parts[0] if parts else ""
            if args.proteins and not any(top.startswith(p + "_") for p in args.proteins):
                return False
            if args.fitness_models and not any(f"_{fm}" in top for fm in args.fitness_models):
                return False
            return True

        dirs = [d for d in all_dirs if _matches(d)]
        print(f"  Found {len(dirs)} result directories.\n")

    if not dirs:
        print("No results directories found. Exiting.")
        import sys
        sys.exit(0)

    # ── Evaluate each directory ──────────────────────────────────────────────
    rows: list[dict] = []
    _per_pos_files: dict[tuple[str, str], list[Path]] = {}

    for d in dirs:
        print(f"Evaluating: {d.name}")

        _ig_files = list(d.glob("*_integrated_gradients_results.csv"))
        _has_splits = False
        if _ig_files:
            try:
                _sample = pd.read_csv(_ig_files[0], usecols=["sequence_name"], nrows=500)
                _tags = _sample["sequence_name"].apply(_extract_split_tag).unique().tolist()
                _has_splits = any(t in _tags for t in ("trainval", "test"))
            except Exception:
                pass

        _split_filters: list[str | None] = [None]
        if _has_splits:
            _split_filters += ["trainval", "test"]

        for split_filter in _split_filters:
            row = evaluate_one(d, use_per_variant=args.use_per_variant,
                               split_filter=split_filter)
            if row is None:
                continue
            rows.append(row)

            _split_label = split_filter or "all"
            if args.per_position_plots and (split_filter is None or args.use_per_variant):
                try:
                    protein = row["protein"]
                    gt = load_ground_truth(protein)
                    L = gt["sequence_length"]
                    if args.use_per_variant:
                        pos_df, _ = _load_per_variant_gradients_robust(
                            d, split_filter=split_filter,
                            min_output_diff=args.min_output_diff,
                        )
                    else:
                        pos_df = _load_position_importance(d)
                    if pos_df is not None:
                        ig_vec = np.zeros(L, dtype=float)
                        for _, r in pos_df.iterrows():
                            p0 = int(r["position"]) - 1
                            if 0 <= p0 < L:
                                ig_vec[p0] = float(r["mean_abs_gradient"])
                        prot_model  = row.get("protein", "") + "_" + row.get("fitness_model", "")
                        noise_part  = d.parent.name if d.parent.name.startswith("noise_") else ""
                        per_pos_lbl = "_".join(filter(None, [prot_model, noise_part, d.name, _split_label]))
                        plot_per_position_comparison(d, protein, gt, ig_vec, output_dir,
                                                     label=per_pos_lbl)
                        _fm  = row.get("fitness_model", "unknown")
                        _itp = row.get("interp", d.name) or d.name
                        _key = (_fm, _itp)
                        _saved_png = output_dir / f"{re.sub(r'[^a-zA-Z0-9_-]', '_', per_pos_lbl)}_per_position.png"
                        _per_pos_files.setdefault(_key, []).append(_saved_png)
                except Exception as exc:
                    print(f"  [warn] Per-position plot failed: {exc}")

            collapse_frac = row.get("collapse_fraction", float("nan"))
            n_reliable    = row.get("n_reliable_positions", "n/a")
            within_bs     = row.get("within_binding_site_spearman", float("nan"))
            signed_rho    = row.get("spearman_r_signed", float("nan"))
            print(
                f"  [{_split_label}]  "
                f"ρ(GT)={row['spearman_r']:+.3f}  τ(GT)={row['kendall_tau']:+.3f}  "
                f"AUROC(GT)={row['auroc']:.3f}  AUPRC(GT)={row['auprc']:.3f}  "
                f"ρ(signed)={signed_rho:+.3f}  "
                f"ρ(fit)={row.get('fitness_spearman_r', float('nan')):+.3f}  "
                f"collapse={collapse_frac:.2f}  "
                f"reliable_pos={n_reliable}  "
                f"ρ(within_bs)={within_bs:+.3f}  "
                f"completeness(5%)={row.get('completeness_frac_pass_5pct', float('nan')):.2f}"
            )

    if not rows:
        print("No valid results found.")
        import sys
        sys.exit(0)

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(rows)
    export_df  = summary_df.drop(
        columns=[c for c in ["_cliff_rows", "_cliff_threshold"] if c in summary_df.columns]
    )
    col_order = [
        "dirname", "protein", "fitness_model", "ig_method", "attribution", "interp",
        "noise_level", "split",
        # Core rank metrics (after reliability mask)
        "spearman_r", "spearman_p", "kendall_tau", "kendall_p",
        "auroc", "auprc", "n_positions", "n_annotated",
        "n_reliable_positions",      # ← fix 4
        "precision_at_k", "precision_at_k_random", "precision_at_k_K",
        # Normalized rank metrics
        "norm_spearman_r", "norm_kendall_tau", "norm_auroc", "norm_auprc",
        # Completeness (with near-zero penalised)
        "completeness_mean_rel_error", "completeness_median_rel_error",
        "completeness_frac_pass_1pct", "completeness_frac_pass_5pct",
        "completeness_frac_pass_10pct", "n_completeness_sequences",
        "n_near_zero_sequences",
        # Collapse diagnostics  ← fix 3
        "collapse_fraction", "n_total_sequences", "n_collapsed_seqs",
        # Fitness-based metrics
        "fitness_spearman_r", "fitness_spearman_p",
        "fitness_kendall_tau", "fitness_kendall_p",
        "fitness_auroc", "fitness_auprc", "n_fitness_positions",
        "fitness_precision_at_k", "fitness_precision_at_k_random",
        # Per-variant GT correlation
        "pv_gt_spearman_mean", "pv_gt_spearman_std", "pv_gt_spearman_median",
        "pv_gt_kendall_mean", "pv_gt_kendall_std", "pv_gt_kendall_median",
        "n_pv_gt_sequences",
        # Per-variant fitness-contribution correlation
        "pvfc_pooled_spearman", "pvfc_pooled_pearson",
        "pvfc_per_seq_spearman_mean", "pvfc_per_seq_spearman_std",
        "pvfc_per_seq_spearman_median", "pvfc_n_pairs", "pvfc_n_sequences",
        # Within-category discrimination (fix 5)
        "within_binding_site_spearman", "within_structural_spearman",
        "within_folding_spearman",
        # Signed IG duplicates (mean raw IG, not abs)
        "spearman_r_signed", "spearman_p_signed",
        "kendall_tau_signed", "kendall_p_signed",
        "auroc_signed", "auprc_signed",
        "precision_at_k_signed", "precision_at_k_random_signed",
        "n_reliable_positions_signed",
        "within_binding_site_spearman_signed",
        "within_structural_spearman_signed",
        "within_folding_spearman_signed",
        # Sign accuracy
        "sign_acc_active", "sign_acc_wt", "sign_acc_overall",
        "n_active_mut", "n_wt_positions",
        # Linear proportionality
        "linear_spearman_r", "linear_pearson_r", "n_prop_obs",
        # Epistatic recovery
        "epi_interacting_mean_absig", "epi_noninteracting_mean_absig",
        "epi_delta_absig", "n_epi_pairs",
    ]
    col_order = [c for c in col_order if c in export_df.columns]
    export_df  = export_df[col_order]

    csv_path = output_dir / "synthetic_ig_validation_summary.csv"
    export_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSummary CSV: {csv_path}")

    # ── Console table ────────────────────────────────────────────────────────
    hdr_cols = [
        ("Directory",     "dirname",                  40, ""),
        ("ρ(GT)",        "spearman_r",                7, "+.3f"),
        ("τ(GT)",        "kendall_tau",               7, "+.3f"),
        ("AUROC(GT)",    "auroc",                     8,  ".3f"),
        ("AUPRC(GT)",    "auprc",                     8,  ".3f"),
        ("Rel.Pos",      "n_reliable_positions",      7,  "d"),
        ("Collapse",     "collapse_fraction",          8,  ".3f"),
        ("ρ(pv-GT)",    "pv_gt_spearman_mean",        9, "+.3f"),
        ("ρ(fit)",      "fitness_spearman_r",          7, "+.3f"),
        ("AUROC(fit)",  "fitness_auroc",               9,  ".3f"),
        ("Cmpl(5%)",    "completeness_frac_pass_5pct",    9,  ".3f"),
        ("ρ(within_bs)","within_binding_site_spearman",   11, "+.3f"),
        ("ρ(within_st)","within_structural_spearman",     11, "+.3f"),
        ("ρ(within_fo)","within_folding_spearman",        11, "+.3f"),
        ("SignAcc(act)","sign_acc_active",                10,  ".3f"),
        ("SignAcc(all)","sign_acc_overall",               10,  ".3f"),
    ]
    hdr_cols = [(h, col, w, fmt) for h, col, w, fmt in hdr_cols if col in export_df.columns]
    header = "".join(f"{h:>{w}}  " for h, col, w, fmt in hdr_cols)
    print("\n" + "=" * len(header))
    print(header.replace("Directory", f"{'Directory':<40}", 1))
    print("-" * len(header))
    for _, row in export_df.iterrows():
        line = f"{str(row['dirname']):<40}  "
        for h, col, w, fmt in hdr_cols[1:]:
            val = row.get(col, float("nan"))
            try:
                if fmt == "d":
                    cell = format(int(val), "d")
                else:
                    fval = float(val)
                    cell = format(fval, fmt) if np.isfinite(fval) else "nan  "
            except (TypeError, ValueError):
                cell = "nan  "
            line += f"{cell:>{w}}  "
        print(line)
    print("=" * len(header))

    # ── Signed-IG df helper: rename *_signed columns to standard names ─────────
    _SIGNED_RENAME: dict[str, str] = {
        "spearman_r_signed":    "spearman_r",
        "spearman_p_signed":    "spearman_p",
        "kendall_tau_signed":   "kendall_tau",
        "kendall_p_signed":     "kendall_p",
        "auroc_signed":         "auroc",
        "auprc_signed":         "auprc",
        "precision_at_k_signed":         "precision_at_k",
        "precision_at_k_random_signed":  "precision_at_k_random",
        "n_reliable_positions_signed":   "n_reliable_positions",
        "within_binding_site_spearman_signed": "within_binding_site_spearman",
        "within_structural_spearman_signed":   "within_structural_spearman",
        "within_folding_spearman_signed":      "within_folding_spearman",
    }

    def _make_signed_df(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with signed-IG metric columns replacing their abs counterparts."""
        d = df.copy()
        # Drop abs versions of columns that have signed counterparts
        d = d.drop(columns=[v for v in _SIGNED_RENAME.values() if v in d.columns],
                   errors="ignore")
        # Rename signed versions to standard metric names
        d = d.rename(columns={k: v for k, v in _SIGNED_RENAME.items() if k in d.columns})
        return d

    # ── Plots (abs IG and signed IG) ──────────────────────────────────────────
    proteins_present = list(export_df["protein"].unique()) if "protein" in export_df.columns else []
    splits_present   = sorted(export_df["split"].unique().tolist()) if "split" in export_df.columns else ["all"]

    def _safe_plots(df_sub: pd.DataFrame, prefix: str) -> None:
        if df_sub.empty:
            return
        for fn, label in [
            (plot_metric_comparison,         "metric comparison"),
            (plot_fitness_metric_comparison, "fitness metric comparison"),
            (_plot_precision_at_k,           "precision@K"),
            (plot_completeness_summary,      "completeness summary"),
        ]:
            try:
                fn(df_sub, output_dir, protein_prefix=prefix)
            except Exception as exc:
                print(f"  [warn] {label} [{prefix}] failed: {exc}")
        try:
            sign_rows = [
                {"label": r["dirname"],
                 "noise_level":      r.get("noise_level",      float("nan")),
                 "interp":           r.get("interp",           ""),
                 "fitness_model":    r.get("fitness_model",    ""),
                 "sign_acc_active":  r.get("sign_acc_active",  float("nan")),
                 "sign_acc_wt":      r.get("sign_acc_wt",      float("nan")),
                 "sign_acc_overall": r.get("sign_acc_overall", float("nan"))}
                for _, r in df_sub.iterrows()
            ]
            plot_sign_accuracy_bars(sign_rows, output_dir, protein_prefix=prefix)
        except Exception as exc:
            print(f"  [warn] Sign accuracy [{prefix}] failed: {exc}")

    fitness_models_present = (
        list(export_df["fitness_model"].unique())
        if "fitness_model" in export_df.columns else []
    )
    for prot in proteins_present:
        prot_df = export_df[export_df["protein"] == prot]
        for split in splits_present:
            split_df = prot_df[prot_df["split"] == split] if "split" in prot_df.columns else prot_df
            if split_df.empty:
                continue
            prefix = f"{prot}_{split}" if split != "all" else prot
            print(f"\n  Generating plots for protein={prot} split={split} ({len(split_df)} runs)")
            _safe_plots(split_df, prefix)
            _safe_plots(_make_signed_df(split_df), prefix + "_signed")

            for fm in fitness_models_present:
                fm_df = (split_df[split_df["fitness_model"] == fm]
                         if "fitness_model" in split_df.columns else pd.DataFrame())
                if fm_df.empty or "noise_level" not in fm_df.columns or fm_df["noise_level"].isna().all():
                    continue
                fm_prefix = f"{prefix}_{fm}" if split != "all" else f"{prot}_{fm}"
                try:
                    plot_noise_vs_interp(fm_df, output_dir, protein=prot, fitness_model=fm,
                                         protein_prefix=fm_prefix, split=split)
                    plot_noise_vs_interp_boxplots(fm_df, output_dir, protein=prot, fitness_model=fm,
                                                   protein_prefix=fm_prefix, split=split)
                except Exception as exc:
                    print(f"  [warn] Noise-vs-interp [{fm_prefix}] failed: {exc}")
                # Signed IG noise plots
                signed_fm_df = _make_signed_df(fm_df)
                signed_fm_prefix = fm_prefix + "_signed"
                try:
                    plot_noise_vs_interp(signed_fm_df, output_dir, protein=prot, fitness_model=fm,
                                         protein_prefix=signed_fm_prefix, split=split)
                    plot_noise_vs_interp_boxplots(signed_fm_df, output_dir, protein=prot, fitness_model=fm,
                                                   protein_prefix=signed_fm_prefix, split=split)
                except Exception as exc:
                    print(f"  [warn] Noise-vs-interp signed [{signed_fm_prefix}] failed: {exc}")

    for split in splits_present:
        split_df = export_df[export_df["split"] == split] if "split" in export_df.columns else export_df
        if split_df.empty:
            continue
        combined_prefix = f"combined_{split}" if split != "all" else None
        signed_combined_prefix = (f"combined_{split}_signed" if split != "all" else "combined_signed")
        print(f"\n  Generating combined overview plots (split={split}, {len(split_df)} runs)")
        _safe_plots(split_df, combined_prefix)
        _safe_plots(_make_signed_df(split_df), signed_combined_prefix)

    print(f"\nDone. Output saved to: {output_dir}")

    # ── Zip per-position plots ────────────────────────────────────────────────
    if args.per_position_plots and _per_pos_files:
        import zipfile
        print("\nZipping per-position plots by (fitness_model, interp) ...")
        for (fm, itp), png_paths in sorted(_per_pos_files.items()):
            zip_name = f"per_position_{fm}_{itp}.zip"
            zip_path = output_dir / zip_name
            existing = [p for p in png_paths if p.exists()]
            if not existing:
                continue
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(existing):
                    zf.write(p, p.name)
            print(f"  {zip_path.name}  ({len(existing)} files)")
            for p in existing:
                p.unlink()
            print(f"  Deleted {len(existing)} source files")


if __name__ == "__main__":
    main()
