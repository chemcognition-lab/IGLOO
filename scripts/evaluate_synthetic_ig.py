#!/usr/bin/env python3
"""
Synthetic IG Validation Script
================================
Evaluates how well integrated gradients attributions recover the known
ground-truth position importance in the synthetic protein fitness datasets.

Metrics per results directory
-------------------------------
1. Spearman ρ   – rank correlation between |IG| and ground-truth importance
                  across all L positions.
2. Kendall τ    – non-parametric rank correlation (more robust to tied values).
3. AUROC        – treats annotated positions (importance > background) as
                  positives; scores = |IG|.  Higher = better.
4. AUPRC        – area under precision-recall curve (better when positives are
                  sparse, which they typically are).

The script discovers results directories automatically or accepts an explicit
list. It parses the protein name from the directory / config, loads the
matching ground-truth JSON, and writes a summary CSV + bar-plot comparison.

Results directory layout expected
----------------------------------
    results/synthetic/
        {protein}_{fitness_model}/          e.g. GB1_additive/, T4Lysozyme_nonlinear/
            noise_{level}/                  e.g. noise_0.0/, noise_0.1/, …, noise_1.0/
                {interp_method}/            e.g. linear/, slerp/, nearest_neighbor/
                    *_integrated_gradients_results.csv
                    *_position_importance_summary.csv

Usage
-----
    # ── Standard production run (RECOMMENDED) ───────────────────────────────
    # Auto-discovers all results/synthetic/{protein}_{model}/noise_*/interp/
    # directories and evaluates every one.  Run this once all IG SLURM jobs
    # have completed.
    #
    source IGLOO_env/bin/activate
    python scripts/evaluate_synthetic_ig.py --use-per-variant

    # Output: results/synthetic/evaluation/
    #   synthetic_ig_validation_summary.csv  – one row per run, all metrics
    #   *_metrics_comparison.png             – bar charts per metric
    #   *_fitness_metrics_comparison.png
    #   *_completeness_summary.png
    #   *_weighted_pos_acc.png
    #   *_sign_accuracy.png
    #   *_noise_vs_interp_{protein}_{model}.png  (one per protein × fitness-model)

    # ── Filter by protein and/or fitness-model ───────────────────────────────
    # All GB1 results (all noise levels, all interp methods, all fitness models):
    python scripts/evaluate_synthetic_ig.py --use-per-variant --proteins GB1

    # All additive results across all proteins:
    python scripts/evaluate_synthetic_ig.py --use-per-variant --fitness-models additive

    # Combine filters:
    python scripts/evaluate_synthetic_ig.py --use-per-variant \\
        --proteins GB1 --fitness-models additive epistatic

    # ── Evaluate specific directories only ──────────────────────────────────
    python scripts/evaluate_synthetic_ig.py \\
        results/synthetic/GB1_additive/noise_0.0/linear \\
        results/synthetic/GB1_additive/noise_0.0/slerp

    # ── Control output location ──────────────────────────────────────────────
    python scripts/evaluate_synthetic_ig.py \\
        --use-per-variant \\
        --output-dir results/synthetic/evaluation

    # ── Also save per-position |IG| vs ground-truth bar charts ──────────────
    python scripts/evaluate_synthetic_ig.py --use-per-variant --per-position-plots

    # ── Flag reference ───────────────────────────────────────────────────────
    # --use-per-variant   Use per-variant integrated_gradients_results.csv
    #                     (recommended; more detailed than pre-aggregated summary)
    # --per-position-plots  Save a |IG| vs ground-truth bar chart per run
    # --output-dir PATH   Override default output location

    # -── Regenerate paper results ─────────────────────────────────────────────
    # To reproduce the main paper figures, run with:
    python scripts/evaluate_synthetic_ig.py --use-per-variant --per-position-plots \\
        --proteins GB1 \\
        --fitness-models additive epistatic nonlinear \\
        --output-dir results/synthetic/evaluation/GB1_results

"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


# Lazy import from synthetic_data (only used for advanced metrics)
def _get_synthetic_protein(protein_name: str):
    """Return the ProteinDefinition object for *protein_name*, or None."""
    try:
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from src.igloo.synthetic_data import PROTEINS as _SP
        return _SP.get(protein_name)
    except Exception:
        return None


def _blosum_penalty(wt_aa: str, mut_aa: str) -> float:
    """BLOSUM62 penalty in [0, 1]; 0 = synonymous, 1 = worst substitution."""
    try:
        from src.igloo.synthetic_data import blosum_penalty as _bp
        return _bp(wt_aa, mut_aa)
    except Exception:
        return float('nan')


def _contact_weight(protein_def, pos1: int, pos2: int) -> float:
    """Sequence-based contact weight proxy (same-SS = 1.0, adj-SS = 0.5, local = 0.3)."""
    try:
        from src.igloo.synthetic_data import _contact_weight as _cw
        return _cw(protein_def, pos1, pos2)
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
SYNTHETIC_RESULTS_DIR = PROJECT_ROOT / "results" / "synthetic"

PROTEINS = ["GB1", "T4Lysozyme", "p53"]

# Background importance assigned to unannotated positions in the synthetic
# data generator (see SyntheticDatasetGenerator.ground_truth_importance).
BACKGROUND_IMPORTANCE = 0.05

# Fitness model categories whose mutations drive the stability cliff.
CLIFF_CRITICAL_CATEGORIES = {"active_site", "metal_binding"}


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def load_ground_truth(protein: str) -> dict:
    """Load ground-truth weights JSON for *protein*.

    Returns a dict with:
      importance_vector  : np.ndarray, shape (L,), 0-indexed, float
      annotated_mask     : np.ndarray, shape (L,), bool  (True = annotated)
      annotated_positions: dict  category → list[int]  (0-indexed)
    """
    gt_path = SYNTHETIC_DATA_DIR / f"{protein}_ground_truth_weights.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    with open(gt_path) as f:
        data = json.load(f)

    vec = np.array(data["importance_vector"], dtype=float)
    annotated_mask = vec > BACKGROUND_IMPORTANCE
    return {
        "importance_vector": vec,
        "annotated_mask": annotated_mask,
        "annotated_positions": data.get("annotated_positions", {}),
        "sequence_length": int(data["sequence_length"]),
    }


# ---------------------------------------------------------------------------
# Results loading helpers
# ---------------------------------------------------------------------------

def _parse_protein_from_dir(results_dir: Path) -> str | None:
    """Try to infer protein name from the directory name."""
    dirname = results_dir.name
    for p in sorted(PROTEINS, key=len, reverse=True):  # longest first to avoid prefix clash
        if p.lower() in dirname.lower():
            return p
    return None


def _load_position_importance(results_dir: Path, dataset_name: str | None = None) -> pd.DataFrame | None:
    """Load the aggregated position_importance_summary.csv from *results_dir*.

    Returns a DataFrame with columns: position (1-indexed), mean_abs_gradient,
    importance_score (and others).  Returns None if file missing.
    """
    # Try generic glob first, then explicit name
    candidates = list(results_dir.glob("*_position_importance_summary.csv"))
    if not candidates and dataset_name:
        candidates = [results_dir / f"{dataset_name}_position_importance_summary.csv"]
    if not candidates:
        return None
    path = candidates[0]
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_per_variant_gradients(
    results_dir: Path,
    dataset_name: str | None = None,
    split_filter: str | None = None,
) -> pd.DataFrame | None:
    """Load the per-variant integrated_gradients_results.csv from *results_dir*.

    If *split_filter* is 'trainval' or 'test', only rows whose sequence_name
    contains the matching ``[trainval]``/``[test]`` tag are included before
    aggregation.

    Returns aggregated (position, mean_abs_gradient) or None.
    """
    candidates = list(results_dir.glob("*_integrated_gradients_results.csv"))
    if not candidates and dataset_name:
        candidates = [results_dir / f"{dataset_name}_integrated_gradients_results.csv"]
    if not candidates:
        return None
    path = candidates[0]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if split_filter and split_filter in ("trainval", "test") and "sequence_name" in df.columns:
        mask = df["sequence_name"].apply(lambda n: _extract_split_tag(n) == split_filter)
        df = df[mask]
        if df.empty:
            return None
    # Aggregate abs_gradient by position across all (filtered) sequences
    agg = (
        df.groupby("position")["abs_gradient"]
        .mean()
        .reset_index()
        .rename(columns={"abs_gradient": "mean_abs_gradient"})
    )
    return agg


def _load_detailed_ig(results_dir: Path) -> pd.DataFrame | None:
    """Load the raw per-sequence per-position IG CSV (includes signed gradients).

    Returns DataFrame with columns sequence_name, position (1-indexed), amino_acid,
    integrated_gradient, abs_gradient, etc.  Returns None if not found.
    """
    candidates = list(results_dir.glob("*_integrated_gradients_results.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def _is_wt_sequence_name(name: str) -> bool:
    """Return True for WT sequence names including tagged variants like 'WT[trainval]'."""
    s = str(name)
    return s == "WT" or s.lower().startswith("wt[")


def _extract_split_tag(name: str) -> str:
    """Extract split label ('trainval', 'test', or 'all') from a sequence name.

    Sequence names produced by run_integrated_gradients.py look like:
        'WT[trainval]', 'W43Y;G38R[test]', 'PolyAlanine', 'seq_0[trainval]', etc.
    Returns 'trainval', 'test', or 'all' (when no tag is present).
    """
    m = re.search(r'\[(trainval|test)\]', str(name), re.IGNORECASE)
    return m.group(1).lower() if m else "all"


def _get_wt_sequence(ig_df: pd.DataFrame) -> str | None:
    """Reconstruct the wildtype sequence from the IG results dataframe.

    Handles both plain 'WT' and tagged names such as 'WT[trainval]'.
    When multiple WT-like entries exist, uses the one with the most position
    rows (most complete).  Returns a 0-indexed string of length L, or None.
    """
    wt_mask = ig_df["sequence_name"].apply(_is_wt_sequence_name)
    wt_df   = ig_df[wt_mask]
    if wt_df.empty:
        return None
    # Pick the WT entry with the most rows (most positional coverage)
    best_name = wt_df["sequence_name"].value_counts().idxmax()
    wt_rows = wt_df[wt_df["sequence_name"] == best_name].sort_values("position")
    return "".join(wt_rows["amino_acid"].tolist())


def _mutations_from_ig(seq_df: pd.DataFrame, wt_seq: str) -> list[tuple[int, str, str]]:
    """Return list of (pos_0indexed, wt_aa, mut_aa) for a single sequence's IG rows.

    seq_df : rows from ig_df filtered to one sequence_name, sorted by position.
    """
    muts = []
    for _, row in seq_df.sort_values("position").iterrows():
        pos_0 = int(row["position"]) - 1
        mut_aa = str(row["amino_acid"])
        if pos_0 < len(wt_seq):
            wt_aa = wt_seq[pos_0]
            if mut_aa != wt_aa:
                muts.append((pos_0, wt_aa, mut_aa))
    return muts


def _load_data_csv(protein: str, fitness_model: str) -> pd.DataFrame | None:
    """Load the original synthetic data CSV for *protein* + *fitness_model*."""
    path = PROJECT_ROOT / "data" / "synthetic" / f"{protein}_{fitness_model}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    ig_scores: np.ndarray,
    gt_importance: np.ndarray,
    annotated_mask: np.ndarray,
) -> dict:
    """Compute all validation metrics for one results directory.

    Parameters
    ----------
    ig_scores      : (L,) array of |IG| values, one per position (0-indexed).
    gt_importance  : (L,) ground-truth importance vector.
    annotated_mask : (L,) bool array; True = functionally annotated position.

    Returns
    -------
    dict with keys: spearman_r, spearman_p, kendall_tau, kendall_p,
                    auroc, auprc, n_positions, n_annotated.
    """
    assert len(ig_scores) == len(gt_importance) == len(annotated_mask), (
        f"Length mismatch: ig={len(ig_scores)}, gt={len(gt_importance)}, "
        f"mask={len(annotated_mask)}"
    )

    # Rank correlations — over all positions
    sp_r, sp_p = spearmanr(ig_scores, gt_importance)
    kt_tau, kt_p = kendalltau(ig_scores, gt_importance)

    # Classification metrics — annotated vs unannotated
    y_true = annotated_mask.astype(int)
    y_score = ig_scores

    # Guard: need at least one positive and one negative
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    # Precision@K: K = n_annotated
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
    }


# ---------------------------------------------------------------------------
# Fitness-based metrics (position-level: |IG| vs mean fitness effect)
# ---------------------------------------------------------------------------

def compute_fitness_based_metrics(
    ig_vec: np.ndarray,
    protein: str,
    fitness_model: str,
    wt_seq: str | None,
) -> dict:
    """Correlate |IG| per position with the mean fitness effect of mutations
    at that position across the full synthetic dataset.

    For each position i we compute:
        mean_fitness_effect[i] = mean( fitness(seq) - fitness(WT) )
                                  over all sequences in the CSV that carry a
                                  mutation at position i.

    We then correlate *ig_vec* (|IG| per position) with
    *-mean_fitness_effect* (so that "more deleterious" = higher value, matching
    the expectation that high |IG| ↔ high importance / deleterious mutations).

    AUROC / AUPRC treat the most deleterious half of **observed** positions as
    positives (score = |IG|).

    Returns
    -------
    dict with keys:
      fitness_spearman_r, fitness_spearman_p,
      fitness_kendall_tau, fitness_kendall_p,
      fitness_auroc, fitness_auprc,
      n_fitness_positions  – number of positions with ≥1 observed mutation
    """
    _nan = {
        "fitness_spearman_r":            float("nan"),
        "fitness_spearman_p":            float("nan"),
        "fitness_kendall_tau":           float("nan"),
        "fitness_kendall_p":             float("nan"),
        "fitness_auroc":                 float("nan"),
        "fitness_auprc":                 float("nan"),
        "n_fitness_positions":           0,
        "fitness_precision_at_k":        float("nan"),
        "fitness_precision_at_k_random": float("nan"),
    }

    data_df = _load_data_csv(protein, fitness_model)
    if data_df is None:
        return _nan
    if "SEQ" not in data_df.columns or "fitness" not in data_df.columns:
        return _nan
    if wt_seq is None:
        return _nan

    L = len(ig_vec)

    # WT fitness: the row whose sequence exactly matches wt_seq
    wt_rows = data_df[data_df["SEQ"] == wt_seq]
    if wt_rows.empty:
        # Fall back: row with no mutations (mutation_string empty / NaN)
        for col in ("mutation_string", "mutations", "num_mutations"):
            if col in data_df.columns:
                wt_rows = data_df[
                    data_df[col].isna() | (data_df[col].astype(str).isin(["", "WT", "wt", "0"]))
                ]
                if not wt_rows.empty:
                    break
    wt_fitness = float(wt_rows["fitness"].iloc[0]) if not wt_rows.empty else float(data_df["fitness"].max())

    # Accumulate fitness effects per position
    pos_effects: list[list[float]] = [[] for _ in range(L)]
    for seq_str, fit in zip(data_df["SEQ"], data_df["fitness"]):
        seq_str = str(seq_str)
        if len(seq_str) != len(wt_seq):
            continue
        delta = float(fit) - wt_fitness
        for i, (s, w) in enumerate(zip(seq_str, wt_seq)):
            if i >= L:
                break
            if s != w:
                pos_effects[i].append(delta)

    # Mean fitness effect per position; NaN where no mutations observed
    mean_effect = np.array([
        float(np.mean(v)) if v else float("nan")
        for v in pos_effects
    ])

    # Only positions with at least one observed mutation
    observed = np.isfinite(mean_effect)
    n_obs = int(observed.sum())
    if n_obs < 3:
        return {**_nan, "n_fitness_positions": n_obs}

    ig_obs  = ig_vec[observed]
    eff_obs = mean_effect[observed]   # typically negative (mutations harmful)

    # Positive correlation: high |IG| ↔ large fitness drop → use -effect
    harm_obs = -eff_obs

    sp_r, sp_p   = spearmanr(ig_obs, harm_obs)
    kt_tau, kt_p = kendalltau(ig_obs, harm_obs)

    # AUROC/AUPRC: top-50% most deleterious positions as positives
    threshold = np.median(eff_obs)
    y_true  = (eff_obs < threshold).astype(int)   # below median = more harmful
    y_score = ig_obs

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    # Fitness Precision@K: K = number of "positive" positions (below median)
    K_fit = int(y_true.sum())
    if K_fit > 0 and K_fit < len(y_score):
        top_k_fit = np.argsort(-y_score)[:K_fit]
        fitness_precision_at_k = float(y_true[top_k_fit].mean())
        fitness_precision_at_k_random = K_fit / len(y_score)
    else:
        fitness_precision_at_k = float("nan")
        fitness_precision_at_k_random = float("nan")

    return {
        "fitness_spearman_r":          float(sp_r),
        "fitness_spearman_p":          float(sp_p),
        "fitness_kendall_tau":         float(kt_tau),
        "fitness_kendall_p":           float(kt_p),
        "fitness_auroc":               float(auroc),
        "fitness_auprc":               float(auprc),
        "n_fitness_positions":         n_obs,
        "fitness_precision_at_k":      fitness_precision_at_k,
        "fitness_precision_at_k_random": fitness_precision_at_k_random,
    }


# ---------------------------------------------------------------------------
# Sign-correctness metrics
# ---------------------------------------------------------------------------

def compute_sign_correctness(
    ig_df: pd.DataFrame,
    gt: dict,
    wt_seq: str,
) -> dict:
    """Compute sign-accuracy metrics for all analyzed (non-WT) sequences.

    Expected signs
    --------------
    - Mutated position that is annotated (critical)  → IG < 0  (mutation is harmful)
    - Mutated position that is unannotated           → IG < 0  (weak background signal)
    - Position with WT residue                        → IG ≥ 0  (no change from baseline)

    Returns
    -------
    dict with:
      sign_acc_active   – fraction of (mutated & annotated) positions with IG < 0
      sign_acc_wt       – fraction of (WT residue) positions with IG ≥ 0
      sign_acc_overall  – fraction of ALL positions with the correct predicted sign
      n_active_mut      – number of (mutated & annotated) position-records
      n_wt_positions    – number of (WT residue) position-records
    """
    annotated_mask = gt["annotated_mask"]
    L = gt["sequence_length"]

    active_correct = 0
    active_total   = 0
    wt_correct     = 0
    wt_total       = 0
    all_correct    = 0
    all_total      = 0

    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        # Per-sequence IG threshold: median |IG| across all positions
        abs_ig_vals = seq_df["integrated_gradient"].abs().values
        ig_threshold = float(np.median(abs_ig_vals)) if len(abs_ig_vals) > 0 else 0.0

        for _, row in seq_df.iterrows():
            pos_0  = int(row["position"]) - 1
            ig_val = float(row["integrated_gradient"])
            mut_aa = str(row["amino_acid"])
            wt_aa  = wt_seq[pos_0] if pos_0 < len(wt_seq) else mut_aa

            is_mutated   = (mut_aa != wt_aa)
            is_annotated = bool(annotated_mask[pos_0]) if pos_0 < L else False

            if is_mutated:
                # Both annotated and unannotated mutations should produce negative IG
                # Only count when |IG| exceeds threshold (not noise)
                correct = (ig_val < -ig_threshold)
                all_correct += int(correct)
                all_total   += 1
                if is_annotated:
                    active_correct += int(correct)
                    active_total   += 1
            else:
                # WT residue → expect non-negative / below noise threshold
                correct = (ig_val >= -ig_threshold)
                wt_correct += int(correct)
                wt_total   += 1
                all_correct += int(correct)
                all_total   += 1

    def _safe_div(n, d): return float(n) / d if d > 0 else float("nan")

    return {
        "sign_acc_active":  _safe_div(active_correct, active_total),
        "sign_acc_wt":      _safe_div(wt_correct, wt_total),
        "sign_acc_overall": _safe_div(all_correct, all_total),
        "n_active_mut":     active_total,
        "n_wt_positions":   wt_total,
    }


# ---------------------------------------------------------------------------
# Practical IG interpretation checks
# ---------------------------------------------------------------------------

# Near-zero output-difference threshold: sequences with |F(x) - F(x')| below
# this value are excluded from the completeness error calculation because the
# relative error ε = |Σ IG − diff| / |diff| becomes unreliably large for very
# small denominators.  This does NOT indicate bad attributions — it simply
# means the sequence has a near-WT fitness effect in normalized space.
_NEAR_ZERO_THRESHOLD = 0.01


def compute_completeness_stats(ig_df: pd.DataFrame) -> dict:
    """Check the IG completeness axiom: Σ_i IG_i ≈ F(x) − F(x').

    For each non-WT sequence the relative completeness error is:
        ε = |Σ IG_i − outputDiff| / max(|outputDiff|, 1e-8)

    Sequences with |outputDiff| < _NEAR_ZERO_THRESHOLD are excluded from the
    relative-error calculation (division instability for tiny denominators).
    These are sequences with near-WT fitness in normalized space — the
    completeness axiom was already verified during IG computation.

    Returns
    -------
    dict with:
      completeness_mean_rel_error  – mean ε over included sequences
      completeness_median_rel_error
      completeness_frac_pass_1pct  – fraction with ε < 0.01
      completeness_frac_pass_5pct  – fraction with ε < 0.05
      completeness_frac_pass_10pct – fraction with ε < 0.10
      n_completeness_sequences     – number of sequences included
      n_near_zero_sequences        – sequences excluded (|outputDiff| < threshold;
                                     near-WT fitness, not faulty attributions)
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

    # Prefer the unclipped output difference: IG sums converge to
    # F_unclipped(x) - F_unclipped(x'), not the clipped version.
    # Fall back to effect_vs_baseline for old CSVs that lack the column.
    use_unclipped = "effect_vs_baseline_unclipped" in ig_df.columns
    diff_col = "effect_vs_baseline_unclipped" if use_unclipped else "effect_vs_baseline"

    errors = []
    n_near_zero = 0

    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        output_diff = float(seq_df[diff_col].iloc[0])
        if abs(output_diff) < _NEAR_ZERO_THRESHOLD:
            n_near_zero += 1
            continue   # unstable — skip
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


def compute_normalized_position_scores(
    ig_df: pd.DataFrame,
    L: int,
) -> np.ndarray | None:
    """Compute position-level |IG| normalized by |F(x) − F(x')| per sequence.

    For each non-WT sequence: norm_ig[pos] = IG[pos] / |F(x) - F(x')|.
    Sequences with |F(x) - F(x')| < threshold are excluded (near-zero, unstable).
    The result is the mean of |norm_ig| over all stable sequences, per position.

    Returns (L,) array, or None if no stable sequences.
    """
    need_cols = {"integrated_gradient", "effect_vs_baseline", "position"}
    if not need_cols.issubset(ig_df.columns):
        return None

    accumulator = np.zeros(L, dtype=float)
    count       = np.zeros(L, dtype=int)

    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        output_diff = float(seq_df["effect_vs_baseline"].iloc[0])
        if abs(output_diff) < _NEAR_ZERO_THRESHOLD:
            continue   # unstable — skip
        for _, row in seq_df.iterrows():
            pos_0 = int(row["position"]) - 1
            if 0 <= pos_0 < L:
                norm_val = abs(float(row["integrated_gradient"])) / abs(output_diff)
                accumulator[pos_0] += norm_val
                count[pos_0]       += 1

    valid = count > 0
    if not valid.any():
        return None
    result = np.where(valid, accumulator / np.maximum(count, 1), 0.0)
    return result

def compute_precision_at_k(
    ig_vec: np.ndarray,
    annotated_mask: np.ndarray,
) -> dict:
    """Precision@K: fraction of annotated positions among the top-K |IG| positions.

    K is set to the number of ground-truth annotated positions, so the metric
    matches the true positive density and has a meaningful per-protein random
    baseline of K/L (annotation density).

    Parameters
    ----------
    ig_vec         : (L,) array of IG values (signed or unsigned).
    annotated_mask : (L,) bool array; True = functionally annotated.

    Returns
    -------
    dict with keys:
      precision_at_k        – fraction of top-K positions that are annotated (0–1)
      precision_at_k_random – random baseline = K/L (annotation density)
      precision_at_k_K      – K used (= n_annotated)
    """
    K = int(annotated_mask.sum())
    L = len(ig_vec)
    if K == 0 or K >= L:
        return {
            "precision_at_k":        float("nan"),
            "precision_at_k_random": float("nan"),
            "precision_at_k_K":      K,
        }
    top_k_idx = np.argsort(-np.abs(ig_vec))[:K]
    precision = float(annotated_mask[top_k_idx].mean())
    random_baseline = K / L
    return {
        "precision_at_k":        precision,
        "precision_at_k_random": random_baseline,
        "precision_at_k_K":      K,
    }


# ---------------------------------------------------------------------------
# Linear-model proportionality metric
# ---------------------------------------------------------------------------

def compute_linear_proportionality(
    ig_df: pd.DataFrame,
    gt: dict,
    wt_seq: str,
) -> dict:
    """For *additive* model: check IG ∝ -(importance × BLOSUM_penalty) at mutated
    annotated positions across all analyzed sequences.

    Returns dict with spearman_proportionality, pearson_proportionality,
    n_mutated_annotated_obs, or NaNs if insufficient data.
    """
    importance_vec  = gt["importance_vector"]
    L               = gt["sequence_length"]

    ig_vals   = []
    exp_vals  = []  # expected ∝ -(importance * BLOSUM_penalty)

    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        for _, row in seq_df.iterrows():
            pos_0  = int(row["position"]) - 1
            mut_aa = str(row["amino_acid"])
            if pos_0 >= L:
                continue
            wt_aa = wt_seq[pos_0]
            if mut_aa == wt_aa:
                continue  # WT position — skip
            importance = float(importance_vec[pos_0])
            penalty    = _blosum_penalty(wt_aa, mut_aa)
            if not np.isfinite(penalty):
                continue
            ig_vals.append(float(row["integrated_gradient"]))
            exp_vals.append(-(importance * penalty))

    if len(ig_vals) < 3:
        return {
            "linear_spearman_r": float("nan"),
            "linear_pearson_r":  float("nan"),
            "n_prop_obs":        len(ig_vals),
        }

    sp_r, _ = spearmanr(ig_vals, exp_vals)
    pe_r, _ = pearsonr(ig_vals, exp_vals)
    return {
        "linear_spearman_r": float(sp_r),
        "linear_pearson_r":  float(pe_r),
        "n_prop_obs":        len(ig_vals),
    }


# ---------------------------------------------------------------------------
# Epistatic pairwise interaction recovery
# ---------------------------------------------------------------------------

def compute_epistatic_recovery(
    ig_df: pd.DataFrame,
    gt: dict,
    protein_name: str,
) -> dict:
    """For *epistatic* model: check whether |IG| is elevated at position pairs
    with non-zero contact weight compared to non-interacting annotated pairs.

    For each analyzed sequence that carries mutations at ≥2 annotated positions,
    we split the mutated annotated pairs into 'interacting' (contact_weight > 0)
    and 'non-interacting' (contact_weight == 0) and compare mean |IG|.

    Returns:
      epi_interacting_mean_absig   – mean |IG| over (mutated, annotated, interacting pair)
      epi_noninteracting_mean_absig– mean |IG| over (mutated, annotated, non-interacting)
      epi_delta_absig              – difference (higher = better recovery)
      n_epi_pairs                  – total (sequence, pair) records evaluated
    """
    protein_def = _get_synthetic_protein(protein_name)
    if protein_def is None:
        return {
            "epi_interacting_mean_absig":    float("nan"),
            "epi_noninteracting_mean_absig": float("nan"),
            "epi_delta_absig":               float("nan"),
            "n_epi_pairs":                   0,
        }

    annotated_mask = gt["annotated_mask"]
    L              = gt["sequence_length"]
    wt_seq         = _get_wt_sequence(ig_df)
    if wt_seq is None:
        return {
            "epi_interacting_mean_absig":    float("nan"),
            "epi_noninteracting_mean_absig": float("nan"),
            "epi_delta_absig":               float("nan"),
            "n_epi_pairs":                   0,
        }

    interacting_absigs     = []
    non_interacting_absigs = []

    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        # Positions that are mutated AND annotated in this sequence
        mut_ann_positions: list[int] = []   # 0-indexed
        pos_absig: dict[int, float] = {}

        for _, row in seq_df.iterrows():
            pos_0  = int(row["position"]) - 1
            mut_aa = str(row["amino_acid"])
            if pos_0 >= L or pos_0 >= len(wt_seq):
                continue
            wt_aa = wt_seq[pos_0]
            is_mutated   = (mut_aa != wt_aa)
            is_annotated = bool(annotated_mask[pos_0])
            if is_mutated and is_annotated:
                mut_ann_positions.append(pos_0)
                pos_absig[pos_0] = abs(float(row["integrated_gradient"]))

        if len(mut_ann_positions) < 2:
            continue

        # Evaluate all pairs of mutated annotated positions
        for ki in range(len(mut_ann_positions)):
            for kj in range(ki + 1, len(mut_ann_positions)):
                p1, p2 = mut_ann_positions[ki], mut_ann_positions[kj]
                cw = _contact_weight(protein_def, p1, p2)
                mean_absig = (pos_absig[p1] + pos_absig[p2]) / 2
                if cw > 0:
                    interacting_absigs.append(mean_absig)
                else:
                    non_interacting_absigs.append(mean_absig)

    def _safe_mean(lst): return float(np.mean(lst)) if lst else float("nan")
    inter_mean = _safe_mean(interacting_absigs)
    nointer_mean = _safe_mean(non_interacting_absigs)
    return {
        "epi_interacting_mean_absig":    inter_mean,
        "epi_noninteracting_mean_absig": nointer_mean,
        "epi_delta_absig":               inter_mean - nointer_mean if np.isfinite(inter_mean) and np.isfinite(nointer_mean) else float("nan"),
        "n_epi_pairs":                   len(interacting_absigs) + len(non_interacting_absigs),
    }


# ---------------------------------------------------------------------------
# Stability-cliff analysis (nonlinear model)
# ---------------------------------------------------------------------------

def compute_cliff_analysis(
    ig_df: pd.DataFrame,
    gt: dict,
    protein_name: str,
    data_df: pd.DataFrame,
) -> dict:
    """For *nonlinear* model: report mean |IG| at cliff-critical positions per
    analyzed sequence, alongside that sequence's cumulative critical damage.

    Returns a dict with:
      cliff_rows : list[dict]  — one per analyzed sequence with
                                 {seq_name, cumulative_damage, mean_absig_critical}
      cliff_threshold : float
    """  # noqa: D401
    protein_def = _get_synthetic_protein(protein_name)
    CLIFF_THRESHOLD = 1.5  # matches fitness_nonlinear default

    annotations = gt["annotated_positions"]  # category → list[int] 0-indexed
    critical_pos: set[int] = set()
    for cat, positions in annotations.items():
        if cat in CLIFF_CRITICAL_CATEGORIES:
            critical_pos.update(positions)

    wt_seq = _get_wt_sequence(ig_df)
    if wt_seq is None or not critical_pos:
        return {"cliff_rows": [], "cliff_threshold": CLIFF_THRESHOLD}

    # Build a lookup: sequence → mutations from data_df (optional, for accuracy)
    seq_to_muts: dict[str, str] = {}
    if data_df is not None and "SEQ" in data_df.columns and "mutation_string" in data_df.columns:
        seq_to_muts = dict(zip(data_df["SEQ"], data_df["mutation_string"]))

    cliff_rows = []
    for seq_name, seq_df in ig_df.groupby("sequence_name"):
        if _is_wt_sequence_name(seq_name):
            continue
        # Compute cumulative critical damage for this sequence
        cumulative_damage = 0.0
        mean_absig_rows   = []

        for _, row in seq_df.iterrows():
            pos_0  = int(row["position"]) - 1
            mut_aa = str(row["amino_acid"])
            if pos_0 >= len(wt_seq):
                continue
            wt_aa = wt_seq[pos_0]
            if mut_aa == wt_aa:
                continue
            # Importance weight
            imp = float(gt["importance_vector"][pos_0]) if pos_0 < gt["sequence_length"] else 0.0
            penalty = _blosum_penalty(wt_aa, mut_aa)
            if np.isfinite(penalty):
                if pos_0 in critical_pos:
                    cumulative_damage += imp * penalty
                # Collect |IG| at critical positions
                if pos_0 in critical_pos:
                    mean_absig_rows.append(abs(float(row["integrated_gradient"])))

        mean_absig_critical = float(np.mean(mean_absig_rows)) if mean_absig_rows else float("nan")
        cliff_rows.append({
            "seq_name":           seq_name,
            "cumulative_damage":  cumulative_damage,
            "mean_absig_critical": mean_absig_critical,
            "above_cliff":        cumulative_damage >= CLIFF_THRESHOLD,
        })

    return {"cliff_rows": cliff_rows, "cliff_threshold": CLIFF_THRESHOLD}


# ---------------------------------------------------------------------------
# Plotting — sign correctness
# ---------------------------------------------------------------------------

def plot_sign_accuracy_bars(
    sign_rows: list[dict],
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Bar chart of sign accuracy for each results directory.

    sign_rows : list of dicts, each with 'label', 'sign_acc_active',
                'sign_acc_wt', 'sign_acc_overall', and optionally
                'noise_level', 'interp', 'fitness_model'.
    """
    if not sign_rows:
        return

    # Sort rows the same way as other bar charts: fitness_model → interp → noise_level
    import pandas as _pd_sa
    _df = _pd_sa.DataFrame(sign_rows)
    if "fitness_model" not in _df.columns:
        _df["fitness_model"] = _df.get("label", "unknown")
    if "interp" not in _df.columns:
        _df["interp"] = _df.get("label", "unknown")
    _df, fm_groups = _sort_df_and_get_groups(_df)
    sign_rows = _df.to_dict("records")

    acc_act  = [r["sign_acc_active"]  for r in sign_rows]
    acc_wt   = [r["sign_acc_wt"]      for r in sign_rows]
    acc_all  = [r["sign_acc_overall"] for r in sign_rows]

    n     = len(sign_rows)
    width = 0.25
    x_pos = _make_x_positions(n, fm_groups)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5), constrained_layout=True)
    b1 = ax.bar(x_pos - width, acc_act, width, label="Active-site (mutated+annotated)", color="steelblue",  alpha=0.85, edgecolor="k", linewidth=0.4)
    b2 = ax.bar(x_pos,         acc_wt,  width, label="WT residues (unchanged)",         color="darkorange", alpha=0.85, edgecolor="k", linewidth=0.4)
    b3 = ax.bar(x_pos + width, acc_all, width, label="Overall",                          color="seagreen",   alpha=0.85, edgecolor="k", linewidth=0.4)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.axhline(0.5, color="grey", lw=1.0, linestyle="--", label="chance (0.5)")
    ax.set_xticks([])
    ax.set_ylabel("Sign accuracy (fraction correct)")
    ax.set_ylim(0, 1.32)  # extra headroom for labels + group header
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    ax.set_title(f"Sign Correctness{title_prot}: should mutations at critical sites produce negative IG?")
    ax.legend(fontsize=7, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_ig_sign_accuracy.png" if protein_prefix else "synthetic_ig_sign_accuracy.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plotting — cliff damage
# ---------------------------------------------------------------------------

def plot_cliff_damage(
    cliff_data: list[dict],
    output_dir: Path,
    label: str = "",
) -> None:
    """Scatter of mean |IG| at cliff-critical positions vs cumulative damage.

    One point per analyzed sequence. Points above the cliff threshold are
    highlighted.
    """
    if not cliff_data:
        return

    damage  = [r["cumulative_damage"]   for r in cliff_data]
    mean_ig = [r["mean_absig_critical"] for r in cliff_data]
    above   = [r["above_cliff"]         for r in cliff_data]
    names   = [r["seq_name"]            for r in cliff_data]
    threshold = cliff_data[0].get("cliff_threshold", 1.5) if cliff_data else 1.5

    colours = ["firebrick" if a else "steelblue" for a in above]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(damage, mean_ig, c=colours, s=80, edgecolors="k", linewidths=0.5, zorder=3)
    for d, ig, n in zip(damage, mean_ig, names):
        if np.isfinite(ig):
            ax.annotate(n, (d, ig), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axvline(threshold, color="grey", linestyle="--", lw=1.2, label=f"cliff threshold = {threshold}")
    ax.set_xlabel("Cumulative critical damage (importance × BLOSUM penalty)")
    ax.set_ylabel("Mean |IG| at cliff-critical positions")
    ax.set_title(f"{label}\nCliff model: |IG| vs cumulative damage at critical sites")

    legend_patches = [
        mpatches.Patch(color="firebrick",  label="above cliff"),
        mpatches.Patch(color="steelblue",  label="below cliff"),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color="grey", lw=1.2, linestyle="--",
                                                    label=f"cliff threshold={threshold}")],
              fontsize=7)

    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    out_path = output_dir / f"{safe_label}_cliff_damage.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Evaluating one results directory
# ---------------------------------------------------------------------------

def evaluate_one(
    results_dir: Path,
    use_per_variant: bool = False,
    split_filter: str | None = None,
) -> dict | None:
    """Evaluate a single results directory.

    Args:
        results_dir: Path to the results directory.
        use_per_variant: Use per-variant CSV instead of pre-aggregated summary.
        split_filter: If 'trainval' or 'test', restrict per-sequence metrics
            to that split.  Position-level metrics (spearman_r, AUROC, …) are
            only split when *use_per_variant* is True.  Pass None for 'all'.

    Returns a flat dict of metadata + metrics, or None if data is missing.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"  [skip] Not a directory: {results_dir}")
        return None

    # ── Infer protein — search the full path, not just leaf dir name ─────────
    # Path hierarchy: {root}/{Protein}_{model}/noise_{level}/{interp}/
    # Build a combined string from all path components for robust parsing.
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

    # ── Parse metadata from the full path hierarchy ──────────────────────────
    # Leaf dir name (e.g. 'linear', 'slerp', 'nearest_neighbor')
    dirname = results_dir.name
    # Parent dir name (e.g. 'noise_0.0', 'noise_0.3')
    noise_dirname = results_dir.parent.name
    # Grandparent dir name (e.g. 'GB1_additive', 'T4Lysozyme_nonlinear')
    model_dirname = results_dir.parent.parent.name

    # Combine all path components for legacy flat dirs too
    search_str = " ".join([full_path_str])

    fitness_model = next(
        (fm for fm in ["additive", "epistatic", "nonlinear"] if fm in model_dirname.lower() or fm in dirname.lower()),
        "unknown",
    )
    ig_method = next(
        (m for m in ["embedding", "token"] if m in search_str.lower()),
        "unknown",
    )
    attribution = next(
        (a for a in ["signed", "magnitude"] if a in search_str.lower()),
        "unknown",
    )
    # Interpolation path: prefer leaf dir name (e.g. 'linear', 'slerp', 'nearest_neighbor'),
    # fall back to regex search across full path.
    _interp_map = {
        "nearest_neighbor": "nearest_neighbor",
        "slerp":            "slerp",
        "linear":           "linear",
        "guided_ig":        "guided_ig",
    }
    interp = _interp_map.get(dirname.lower(), None)
    if interp is None:
        interp_match = re.search(
            r"(nearest_neighbor|slerp|linear|guided_ig|linearinterpolation)",
            full_path_str,
            re.IGNORECASE,
        )
        interp = interp_match.group(1).lower() if interp_match else "unknown"

    # Noise level: parse from parent dir name, e.g. 'noise_0.05' → 0.05
    noise_match = re.search(r"noise[_\-]([0-9]*\.?[0-9]+)", noise_dirname, re.IGNORECASE)
    noise_level: float | None = float(noise_match.group(1)) if noise_match else None

    # ── Load ground truth ────────────────────────────────────────────────────
    try:
        gt = load_ground_truth(protein)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    L = gt["sequence_length"]

    # ── Load IG position scores ──────────────────────────────────────────────
    if use_per_variant:
        pos_df = _load_per_variant_gradients(results_dir, split_filter=split_filter)
        score_col = "mean_abs_gradient"
    else:
        # Pre-aggregated summary has no split info; only split ig_df-based metrics below
        pos_df = _load_position_importance(results_dir)
        score_col = "mean_abs_gradient"

    if pos_df is None or pos_df.empty:
        print(f"  [skip] No position data found in {results_dir}")
        return None

    # ── Align CSV (1-indexed) to ground-truth (0-indexed, length L) ─────────
    ig_vec = np.zeros(L, dtype=float)
    for _, row in pos_df.iterrows():
        pos_1 = int(row["position"])          # 1-indexed from CSV
        pos_0 = pos_1 - 1                     # 0-indexed for ground truth
        if 0 <= pos_0 < L:
            ig_vec[pos_0] = float(row[score_col])

    gt_vec = gt["importance_vector"]
    mask = gt["annotated_mask"]

    # ── Compute core metrics ─────────────────────────────────────────────────
    metrics = compute_metrics(ig_vec, gt_vec, mask)

    # ── Precision@K: top-K |IG| positions vs annotated mask ─────────────────
    weighted_acc_metrics = compute_precision_at_k(ig_vec, mask)

    # ── Detailed per-sequence IG for advanced metrics ────────────────────────
    ig_df_full = _load_detailed_ig(results_dir)
    # Always reconstruct WT from the full (unfiltered) dataframe
    wt_seq = _get_wt_sequence(ig_df_full) if ig_df_full is not None else None

    # Apply split filter to the per-sequence IG dataframe
    ig_df = ig_df_full
    if ig_df is not None and split_filter and split_filter in ("trainval", "test"):
        mask_split = ig_df["sequence_name"].apply(
            lambda n: _extract_split_tag(n) == split_filter
        )
        ig_df = ig_df[mask_split]
        if ig_df.empty:
            ig_df = None  # nothing in this split

    # 1. Sign correctness
    sign_metrics: dict = {}
    if ig_df is not None and wt_seq is not None:
        sign_metrics = compute_sign_correctness(ig_df, gt, wt_seq)

    # 2. Linear proportionality (always useful, especially for additive model)
    prop_metrics: dict = {}
    if ig_df is not None and wt_seq is not None:
        prop_metrics = compute_linear_proportionality(ig_df, gt, wt_seq)

    # 3. Epistatic pair recovery (only meaningful for epistatic model, but compute anyway)
    epi_metrics: dict = {}
    if ig_df is not None:
        epi_metrics = compute_epistatic_recovery(ig_df, gt, protein)

    # 4. Cliff analysis (most meaningful for nonlinear model)
    cliff_result: dict = {"cliff_rows": [], "cliff_threshold": 1.5}
    if ig_df is not None and wt_seq is not None:
        data_df = _load_data_csv(protein, fitness_model)
        cliff_result = compute_cliff_analysis(ig_df, gt, protein, data_df)

    # 5. Fitness-based metrics — |IG| vs mean fitness effect per position
    fitness_metrics = compute_fitness_based_metrics(ig_vec, protein, fitness_model, wt_seq)

    # ── Practical IG interpretation checks ──────────────────────────────────
    # [1] Completeness axiom: Σ IG_i ≈ F(x) - F(x')
    completeness_metrics: dict = {}
    if ig_df is not None:
        completeness_metrics = compute_completeness_stats(ig_df)

    # [2] Normalized position scores: IG_i / |F(x) - F(x')| for cross-sample comparison
    # Re-run rank-correlation & classification metrics on the normalized scores.
    norm_metrics: dict = {}
    if ig_df is not None:
        norm_ig_vec = compute_normalized_position_scores(ig_df, L)
        if norm_ig_vec is not None:
            raw = compute_metrics(norm_ig_vec, gt_vec, mask)
            norm_metrics = {f"norm_{k}": v for k, v in raw.items()
                            if k in ("spearman_r", "kendall_tau", "auroc", "auprc")}

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
        **completeness_metrics,
        **sign_metrics,
        **prop_metrics,
        **epi_metrics,
        **fitness_metrics,
        "_cliff_rows":      cliff_result["cliff_rows"],
        "_cliff_threshold": cliff_result["cliff_threshold"],
    }


# ---------------------------------------------------------------------------
# Discover results directories
# ---------------------------------------------------------------------------

def discover_synthetic_results(root: Path = SYNTHETIC_RESULTS_DIR) -> list[Path]:
    """Return all leaf subdirectories under *root* that contain IG result CSVs.

    Supports the three-level hierarchy produced by submit_sweep.sh:
      {root}/{Protein}_{model}/noise_{level}/{interp}/
    as well as legacy flat directories directly under *root*.
    """
    import os

    _SUFFIXES = ("_position_importance_summary.csv", "_integrated_gradients_results.csv")

    def _has_csv(d: Path) -> bool:
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False) and any(
                        entry.name.endswith(s) for s in _SUFFIXES
                    ):
                        return True
        except PermissionError:
            pass
        return False

    dirs = []
    for level1 in sorted(root.iterdir()):          # e.g. GB1_additive/
        if not level1.is_dir():
            continue
        if _has_csv(level1):
            dirs.append(level1)
            continue
        for level2 in sorted(level1.iterdir()):    # e.g. noise_0.0/
            if not level2.is_dir():
                continue
            if _has_csv(level2):
                dirs.append(level2)
                continue
            for level3 in sorted(level2.iterdir()):  # e.g. linear/
                if not level3.is_dir():
                    continue
                if _has_csv(level3):
                    dirs.append(level3)
    return dirs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colormaps assigned to each fitness model; shades vary by interpolation path.
_FITNESS_MODEL_CMAPS: dict[str, any] = {
    "additive":  plt.cm.Blues,
    "epistatic": plt.cm.Oranges,
    "nonlinear": plt.cm.Greens,
}
_DEFAULT_CMAP = plt.cm.Purples

# Shade intensity for each interpolation path within a model-family colormap.
_INTERP_SHADES: dict[str, float] = {
    "linear":              0.40,
    "linearinterpolation": 0.40,
    "slerp":               0.58,
    "nearest_neighbor":    0.72,
    "guided_ig":           0.88,
}


def _assign_colors(df: pd.DataFrame) -> list:
    """Return one RGBA colour per row.

    Hue = interpolation path (consistent with noise-vs-interp line plots).
    Lightness = noise level: low noise → light tint, high noise → full saturation.
    Falls back to fitness-model colourmap when no noise_level column is present.
    """
    import matplotlib.colors as mcolors

    # Check whether noise_level varies in this dataframe
    has_noise = (
        "noise_level" in df.columns
        and df["noise_level"].notna().any()
    )
    noise_vals = df["noise_level"].values if has_noise else None
    noise_min  = float(np.nanmin(noise_vals)) if has_noise else 0.0
    noise_max  = float(np.nanmax(noise_vals)) if has_noise else 1.0
    noise_range = max(noise_max - noise_min, 1e-6)

    colours = []
    for i, (_, row) in enumerate(df.iterrows()):
        interp = str(row.get("interp", "unknown"))
        base_hex = _INTERP_COLORS.get(interp, "#607D8B")
        base_rgb = np.array(mcolors.to_rgb(base_hex))

        if has_noise and noise_vals is not None:
            nl = noise_vals[i]
            if np.isfinite(nl):
                # t=0 → full colour (noise=0), t=1 → very light (noise=max)
                t = 1.0 - (float(nl) - noise_min) / noise_range
                t = 0.20 + 0.80 * t   # keep minimum lightness so bars are visible
                colour = tuple(1.0 - t * (1.0 - c) for c in base_rgb) + (1.0,)
                colours.append(colour)
                continue
        # Fallback: use fitness-model colormap as before
        model = str(row.get("fitness_model", "unknown"))
        cmap  = _FITNESS_MODEL_CMAPS.get(model, _DEFAULT_CMAP)
        shade = _INTERP_SHADES.get(interp, 0.55)
        colours.append(cmap(shade))
    return colours


def _config_legend_handles(df: pd.DataFrame) -> list:
    """Legend patches: one per unique interp path (full-saturation colour).

    When noise_level varies, adds a second set of ghost patches showing
    the light→dark scale so the reader understands the shading.
    """
    import matplotlib.colors as mcolors

    has_noise = (
        "noise_level" in df.columns
        and df["noise_level"].notna().any()
        and df["noise_level"].nunique() > 1
    )

    # One full-saturation patch per interp path
    seen: dict[str, mpatches.Patch] = {}
    for _, row in df.iterrows():
        interp = str(row.get("interp", "unknown"))
        if interp not in seen:
            base_hex = _INTERP_COLORS.get(interp, "#607D8B")
            seen[interp] = mpatches.Patch(color=base_hex, label=f"interp: {interp}")
    handles = list(seen.values())

    if has_noise:
        # Light-to-dark gradient hint patches
        handles.append(mpatches.Patch(color="#444444", label="← noise=low (dark)"))
        handles.append(mpatches.Patch(color="#dddddd", label="→ noise=high (light)"))

    return handles


def _sort_df_and_get_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """Sort *df* by fitness_model → interp → noise_level and return group boundaries.

    Returns
    -------
    sorted_df   : reset-indexed copy sorted for grouped display
    fm_groups   : list of (fitness_model_name, start_idx, end_idx_inclusive)
    """
    _fm_order = {"additive": 0, "epistatic": 1, "nonlinear": 2}
    df = df.copy()
    df["_fm_sort"] = df["fitness_model"].map(lambda x: _fm_order.get(str(x).lower(), 99))
    sort_cols = ["_fm_sort", "interp"]
    if "noise_level" in df.columns:
        df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")
        sort_cols.append("noise_level")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.drop(columns=["_fm_sort"], inplace=True)

    fm_vals = df["fitness_model"].values
    n = len(fm_vals)
    groups: list[tuple[str, int, int]] = []
    i_start = 0
    for k in range(1, n):
        if fm_vals[k] != fm_vals[k - 1]:
            groups.append((str(fm_vals[i_start]), i_start, k - 1))
            i_start = k
    groups.append((str(fm_vals[i_start]), i_start, n - 1))
    return df, groups


def _make_x_positions(
    n: int,
    fm_groups: list[tuple[str, int, int]],
    gap: float = 2.0,
) -> np.ndarray:
    """Return x-axis positions for *n* bars with extra *gap* between fitness-model groups.

    Within each group bars are spaced 1 unit apart; *gap* extra units are
    inserted between consecutive groups so they are visually separated.
    """
    x = np.zeros(n)
    cursor = 0.0
    for g_idx, (_, g_start, g_end) in enumerate(fm_groups):
        if g_idx > 0:
            cursor += gap  # extra gap before each new group
        for i in range(g_start, g_end + 1):
            x[i] = cursor
            cursor += 1.0
    return x


def _apply_group_decorations(
    ax,
    fm_groups: list[tuple[str, int, int]],
    x_pos: np.ndarray | None = None,
) -> None:
    """Draw vertical separators and fitness-model header labels on *ax*.

    If *x_pos* is provided it is used for correct placement when groups have
    been given non-uniform x coordinates (i.e. via _make_x_positions).
    """
    y_top = ax.get_ylim()[1]
    for g_idx, (fm_name, g_start, g_end) in enumerate(fm_groups):
        if x_pos is not None:
            mid = (x_pos[g_start] + x_pos[g_end]) / 2
            sep_x = (x_pos[g_start] - 1.0) if g_start > 0 else None  # midpoint between groups
            if g_start > 0 and x_pos is not None:
                sep_x = (x_pos[g_start - 1 if g_start > 0 else g_start] + x_pos[g_start]) / 2
        else:
            mid = (g_start + g_end) / 2
            sep_x = g_start - 0.5 if g_start > 0 else None
        ax.text(mid, y_top * 0.98, fm_name,
                ha="center", va="top", fontsize=7, fontweight="bold", color="dimgray",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="lightgray", alpha=0.7))
        if g_start > 0 and sep_x is not None:
            ax.axvline(sep_x, color="lightgray", lw=1.2, linestyle="-", zorder=0)


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Grouped bar charts — one panel per metric, one bar per configuration."""
    metrics = ["spearman_r", "kendall_tau", "auroc", "auprc", "precision_at_k"]
    metric_labels = {
        "spearman_r":      "Spearman ρ",
        "kendall_tau":     "Kendall τ",
        "auroc":           "AUROC",
        "auprc":           "AUPRC",
        "precision_at_k":  "Precision@K",
    }

    summary_df = summary_df.copy()
    summary_df, fm_groups = _sort_df_and_get_groups(summary_df)

    n_configs = len(summary_df)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(max(15, n_configs * 1.0), 10),
        constrained_layout=True,
    )
    axes = axes.flatten()
    # Hide the 6th panel (only 5 metrics)
    axes[5].set_visible(False)

    colours = _assign_colors(summary_df)
    prot_handles = _config_legend_handles(summary_df)
    x_pos = _make_x_positions(n_configs, fm_groups)

    # Random-chance baselines per metric
    if "n_annotated" in summary_df.columns and "n_positions" in summary_df.columns:
        auprc_baseline = float(
            (summary_df["n_annotated"] / summary_df["n_positions"].replace(0, np.nan)).mean()
        )
    else:
        auprc_baseline = float("nan")
    # Per-run random baseline for Precision@K
    if "precision_at_k_random" in summary_df.columns:
        pak_random_mean = float(summary_df["precision_at_k_random"].mean())
    else:
        pak_random_mean = float("nan")
    _gt_baselines = {
        "spearman_r":     (0.0,            "random (0)"),
        "kendall_tau":    (0.0,            "random (0)"),
        "auroc":          (0.5,            "random (0.5)"),
        "auprc":          (auprc_baseline, f"random ({auprc_baseline:.2f})"),
        "precision_at_k": (pak_random_mean, f"random ({pak_random_mean:.2f})"),
    }

    for panel_i, (ax, metric) in enumerate(zip(axes, metrics)):
        vals = summary_df[metric].values
        bars = ax.bar(
            x_pos, vals,
            color=colours, edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        base_val, base_label = _gt_baselines.get(metric, (float("nan"), ""))
        extra_handles = []
        if np.isfinite(base_val):
            ax.axhline(base_val, color="grey", lw=1.2, linestyle="--")
            extra_handles.append(
                plt.Line2D([0], [0], color="grey", lw=1.2, linestyle="--", label=base_label)
            )
        if panel_i == 0:
            ax.legend(handles=prot_handles + extra_handles, fontsize=7, loc="upper right")
        elif extra_handles:
            ax.legend(handles=extra_handles, fontsize=7)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
        ax.set_xticks([])
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.set_title(metric_labels[metric], fontsize=10, fontweight="bold")
        y_min = min(0, float(np.nanmin(vals)) - 0.05)
        # headroom: enough for rotated value labels (~20% of data range above max)
        y_max = float(np.nanmax(vals))
        y_range = max(y_max - y_min, 0.1)
        ax.set_ylim(bottom=y_min, top=y_max + y_range * 0.28)
        # tight x limits with half-bar padding on each side
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    fig.suptitle(
        f"Synthetic IG Validation{title_prot} — Alignment with Ground-Truth Position Importance\n"
        "(Spearman/Kendall: rank correlation |IG| vs importance; "
        "AUROC/AUPRC: annotated vs unannotated positions)",
        fontsize=9,
    )

    fname = f"{protein_prefix}_ig_validation_metrics.png" if protein_prefix else "synthetic_ig_validation_metrics.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_fitness_metric_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Grouped bar charts comparing |IG| to observed fitness effects.

    Same layout as plot_metric_comparison but uses the fitness_* metric columns.
    """
    fitness_metrics = ["fitness_spearman_r", "fitness_kendall_tau", "fitness_auroc", "fitness_auprc", "fitness_precision_at_k"]
    # Only plot if at least one run has these columns populated
    available = [m for m in fitness_metrics if m in summary_df.columns and summary_df[m].notna().any()]
    if not available:
        print("  [skip] No fitness-based metrics available to plot.")
        return

    metric_labels = {
        "fitness_spearman_r":   "Spearman ρ (fitness)",
        "fitness_kendall_tau":  "Kendall τ (fitness)",
        "fitness_auroc":        "AUROC (fitness)",
        "fitness_auprc":        "AUPRC (fitness)",
        "fitness_precision_at_k": "Precision@K (fitness)",
    }

    summary_df = summary_df.copy()
    summary_df, fm_groups = _sort_df_and_get_groups(summary_df)

    n_configs = len(summary_df)
    n_metrics = len(available)
    ncols = 2
    nrows = (n_metrics + 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(10, n_configs * 0.7), 5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    colours = _assign_colors(summary_df)
    prot_handles = _config_legend_handles(summary_df)
    x_pos = _make_x_positions(n_configs, fm_groups)

    if "fitness_precision_at_k_random" in summary_df.columns:
        fit_pak_random_mean = float(summary_df["fitness_precision_at_k_random"].mean())
    else:
        fit_pak_random_mean = float("nan")
    _fit_baselines = {
        "fitness_spearman_r":      (0.0, "random (0)"),
        "fitness_kendall_tau":     (0.0, "random (0)"),
        "fitness_auroc":           (0.5, "random (0.5)"),
        "fitness_auprc":           (0.5, "random (0.5)"),
        "fitness_precision_at_k":  (fit_pak_random_mean, f"random ({fit_pak_random_mean:.2f})"),
    }

    for i, (ax, metric) in enumerate(zip(axes_flat, available)):
        vals = summary_df[metric].values
        bars = ax.bar(
            x_pos, vals,
            color=colours, edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        base_val, base_label = _fit_baselines.get(metric, (float("nan"), ""))
        extra_handles = []
        if np.isfinite(base_val):
            ax.axhline(base_val, color="grey", lw=1.2, linestyle="--")
            extra_handles.append(
                plt.Line2D([0], [0], color="grey", lw=1.2, linestyle="--", label=base_label)
            )
        if i == 0:
            ax.legend(handles=prot_handles + extra_handles, fontsize=7, loc="upper right")
        elif extra_handles:
            ax.legend(handles=extra_handles, fontsize=7)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
        ax.set_xticks([])
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)
        ax.set_title(metric_labels.get(metric, metric), fontsize=10, fontweight="bold")
        y_min = min(0, float(np.nanmin(vals)) - 0.05)
        y_max = float(np.nanmax(vals))
        y_range = max(y_max - y_min, 0.1)
        ax.set_ylim(bottom=y_min, top=y_max + y_range * 0.28)
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

    for ax in axes_flat[len(available):]:
        ax.set_visible(False)

    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    fig.suptitle(
        f"Synthetic IG Validation{title_prot} — Alignment with Observed Fitness Effects\n"
        "(Spearman/Kendall: rank correlation |IG| vs mean fitness drop per position; "
        "AUROC/AUPRC: top-50% most deleterious positions as positives)",
        fontsize=9,
    )

    fname = f"{protein_prefix}_ig_fitness_metrics.png" if protein_prefix else "synthetic_ig_fitness_metrics.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_completeness_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Bar chart of completeness axiom pass-rates and normalized-IG rank metrics.

    Two sub-plots:
      Left  — fraction of sequences satisfying Σ IG ≈ F(x)−F(x') at 1/5/10% tolerance
      Right — Spearman ρ on raw |IG| vs. on |IG|/|ΔF| (normalized), to show whether
              normalizing improves ground-truth alignment.
    """
    comp_col = "completeness_frac_pass_5pct"  # primary quality gauge
    has_completeness = comp_col in summary_df.columns and summary_df[comp_col].notna().any()
    has_norm = "norm_spearman_r" in summary_df.columns and summary_df["norm_spearman_r"].notna().any()

    if not has_completeness and not has_norm:
        print("  [skip] No completeness / normalized-IG data to plot.")
        return

    df, fm_groups = _sort_df_and_get_groups(summary_df)
    n = len(df)
    colours = _assign_colors(df)
    config_handles = _config_legend_handles(df)
    x_pos = _make_x_positions(n, fm_groups)

    n_panels = int(has_completeness) + int(has_norm)
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(8, n * 0.9), 5 * n_panels),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    panel = 0
    if has_completeness:
        ax = axes[panel]; panel += 1
        tols = [("completeness_frac_pass_1pct",  "ε<1%",  0.55),
                (comp_col,                        "ε<5%",  0.68),
                ("completeness_frac_pass_10pct",  "ε<10%", 0.82)]
        w = 0.22
        offsets = [-w, 0, w]
        bar_colors = [plt.cm.Blues(s) for _, _, s in tols]
        for k, (col, lbl, _) in enumerate(tols):
            vals = df[col].values if col in df.columns else np.full(n, np.nan)
            brs = ax.bar(x_pos + offsets[k], vals, w, color=bar_colors[k],
                         edgecolor="k", linewidth=0.4, alpha=0.85, label=lbl)
            for bar, v in zip(brs, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=5, rotation=90)
        ax.set_xticks([])
        ax.set_ylabel("Fraction of sequences passing")
        ax.set_ylim(0, 1.32)  # extra headroom for rotated labels + group header
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        title_prot = f" [{protein_prefix}]" if protein_prefix else ""
        ax.set_title(f"Completeness Axiom Pass-Rate{title_prot}\n(Σ IG ≈ F(x)−F(x′) within tolerance)")
        ax.legend(handles=config_handles + [mpatches.Patch(color=c, label=l) for c, l in
                  [(plt.cm.Blues(s), lb) for _, lb, s in tols]], fontsize=7, loc="lower right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

        if "n_near_zero_sequences" in df.columns:
            for idx, (xp, (_, row)) in enumerate(zip(x_pos, df.iterrows())):
                nz = int(row.get("n_near_zero_sequences", 0))
                if nz > 0:
                    ax.text(xp, -0.06, f"⚠{nz}", ha="center", va="top",
                            fontsize=6, color="firebrick", transform=ax.transData)

    if has_norm:
        ax = axes[panel]
        raw_vals  = df["spearman_r"].values  if "spearman_r"  in df.columns else np.full(n, np.nan)
        norm_vals = df["norm_spearman_r"].values
        w = 0.35
        ax.bar(x_pos - w/2, raw_vals,  w, color=colours, edgecolor="k",
               linewidth=0.4, alpha=0.85, label="raw |IG|")
        ax.bar(x_pos + w/2, norm_vals, w, color=colours, edgecolor="k",
               linewidth=0.4, alpha=0.55, label="|IG| / |ΔF| (normalised)")
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xticks([])
        ax.set_ylabel("Spearman ρ vs ground-truth importance")
        y_min = min(0, float(np.nanmin(np.concatenate([raw_vals, norm_vals]))) - 0.05)
        y_max = float(np.nanmax(np.concatenate([raw_vals, norm_vals])))
        y_range = max(y_max - y_min, 0.1)
        ax.set_ylim(bottom=y_min, top=y_max + y_range * 0.28)
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        title_prot = f" [{protein_prefix}]" if protein_prefix else ""
        ax.set_title(f"Raw vs Normalised |IG| Rank Correlation{title_prot}\n"
                     "(normalise by |F(x)−F(x′)| to compare across sequences)")
        ax.legend(handles=config_handles, fontsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_completeness.png" if protein_prefix else "synthetic_completeness.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_precision_at_k(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein_prefix: str | None = None,
) -> None:
    """Bar chart of Precision@K per run, with per-run random baseline markers.

    K = number of annotated positions (ground truth), so the random baseline
    equals the annotation density (K/L) and varies per protein/run.
    """
    col = "precision_at_k"
    if col not in summary_df.columns or summary_df[col].isna().all():
        print("  [skip] No precision_at_k data to plot.")
        return

    df, fm_groups = _sort_df_and_get_groups(summary_df)
    n = len(df)

    colours = _assign_colors(df)
    config_handles = _config_legend_handles(df)
    x_pos = _make_x_positions(n, fm_groups)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 5), constrained_layout=True)
    vals = df[col].values
    bars = ax.bar(x_pos, vals, color=colours, edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    # Per-run random baseline (annotation density = K/L, may differ per run)
    if "precision_at_k_random" in df.columns:
        rand_vals = df["precision_at_k_random"].values
        for xp, rv in zip(x_pos, rand_vals):
            if np.isfinite(rv):
                ax.plot(xp, rv, marker="_", color="grey", markersize=12, markeredgewidth=1.5)
        rand_handle = plt.Line2D(
            [0], [0], color="grey", marker="_", linestyle="none",
            markersize=10, markeredgewidth=1.5, label="random (K/L)"
        )
        config_handles = config_handles + [rand_handle]

    ax.set_xticks([])
    ax.set_ylabel("Precision@K  (K = n_annotated)")
    y_max = float(np.nanmax(vals))
    ax.set_ylim(0, min(1.0, max(1.0, y_max)) * 1.28)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    title_prot = f" [{protein_prefix}]" if protein_prefix else ""
    ax.set_title(
        f"Precision@K{title_prot}\n"
        "Fraction of top-K |IG| positions that are annotated; K = n_annotated (grey tick = random baseline)"
    )
    ax.legend(handles=config_handles, fontsize=7, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _apply_group_decorations(ax, fm_groups, x_pos)

    fname = f"{protein_prefix}_precision_at_k.png" if protein_prefix else "synthetic_precision_at_k.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# Distinct colors for each interpolation path
_INTERP_COLORS: dict[str, str] = {
    "linear":           "#2196F3",   # blue
    "slerp":            "#FF5722",   # deep orange
    "nearest_neighbor": "#4CAF50",   # green
    "guided_ig":        "#9C27B0",   # purple
}
_INTERP_MARKERS: dict[str, str] = {
    "linear":           "o",
    "slerp":            "s",
    "nearest_neighbor": "^",
    "guided_ig":        "D",
}


def plot_noise_vs_interp(
    summary_df: pd.DataFrame,
    output_dir: Path,
    protein: str,
    fitness_model: str,
    protein_prefix: str | None = None,
    split: str = "all",
) -> None:
    """Line plots of key metrics vs noise level, coloured by interpolation path.

    For a single (protein, fitness_model) subset, produces a grid of panels —
    one per metric.  Each interpolation path is a separate line so you can
    read off:
      - How does metric degrade as noise increases?
      - Which interp path is most robust to noise?

    Metrics shown: Spearman ρ (GT), AUROC (GT), AUPRC (GT),
                   Spearman ρ (fitness), AUROC (fitness),
                   sign accuracy (active-site), completeness (5% pass-rate),
                   weighted position accuracy.
    """
    if "noise_level" not in summary_df.columns:
        return

    df = summary_df[
        (summary_df["protein"] == protein)
        & (summary_df["fitness_model"] == fitness_model)
    ].copy()
    if df.empty:
        return

    # Drop rows where noise_level is None/NaN
    df = df[df["noise_level"].notna()]
    if df.empty:
        return

    df["noise_level"] = df["noise_level"].astype(float)

    metrics = [
        ("spearman_r",                "Spearman ρ (GT)",         0.0),
        ("auroc",                     "AUROC (GT)",               0.5),
        ("auprc",                     "AUPRC (GT)",               None),
        ("fitness_spearman_r",        "Spearman ρ (fitness)",    0.0),
        ("fitness_auroc",             "AUROC (fitness)",          0.5),
        ("sign_acc_active",           "Sign acc (active-site)",  0.5),
        ("completeness_frac_pass_5pct", "Completeness (5% pass)", None),
        ("precision_at_k",            "Precision@K",             None),
    ]
    # Keep only metrics that actually exist in the dataframe
    metrics = [(col, lbl, base) for col, lbl, base in metrics if col in df.columns]
    if not metrics:
        return

    # Pre-compute near-zero fraction per (interp, noise_level):
    #   frac = n_near_zero / (n_completeness_sequences + n_near_zero)
    # so 0 = fully reliable, 1 = all sequences excluded
    _has_near_zero = (
        "n_near_zero_sequences" in df.columns
        and "n_completeness_sequences" in df.columns
    )

    n_panels = len(metrics)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.5),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    interps = sorted(df["interp"].unique())
    noise_levels = sorted(df["noise_level"].unique())

    for ax_idx, (col, label, baseline) in enumerate(metrics):
        ax = axes_flat[ax_idx]
        for interp in interps:
            sub = df[df["interp"] == interp].sort_values("noise_level")
            if sub.empty:
                continue
            color  = _INTERP_COLORS.get(interp, "#607D8B")
            marker = _INTERP_MARKERS.get(interp, "o")
            vals = sub[col].values.astype(float)
            ax.plot(
                sub["noise_level"].values, vals,
                color=color, marker=marker, linewidth=1.8, markersize=6,
                label=interp, alpha=0.88,
            )

        if baseline is not None:
            ax.axhline(baseline, color="grey", lw=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("Noise level", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(n) for n in noise_levels], fontsize=7, rotation=45)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

        # ── Secondary axis: near-zero fraction ──────────────────────────
        if _has_near_zero:
            ax2 = ax.twinx()
            for interp in interps:
                sub = df[df["interp"] == interp].sort_values("noise_level")
                if sub.empty:
                    continue
                total = sub["n_near_zero_sequences"] + sub["n_completeness_sequences"]
                frac = np.where(
                    total > 0,
                    sub["n_near_zero_sequences"].values / total.values,
                    0.0,
                )
                color = _INTERP_COLORS.get(interp, "#607D8B")
                ax2.plot(
                    sub["noise_level"].values, frac,
                    color=color, linewidth=1.0, linestyle=":",
                    alpha=0.5, marker=None,
                )
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("near-zero frac ↑ = unreliable", fontsize=6, color="dimgrey")
            ax2.tick_params(axis="y", labelsize=6, colors="dimgrey")
            ax2.spines["right"].set_color("dimgrey")
            ax2.spines["right"].set_linestyle(":")
            # Shade background red when near-zero fraction is high (>0.3)
            # Use the mean across interps per noise level for the shading
            mean_frac = (
                df.groupby("noise_level").apply(
                    lambda g: (g["n_near_zero_sequences"].sum()
                               / max((g["n_near_zero_sequences"] + g["n_completeness_sequences"]).sum(), 1))
                ).reindex(noise_levels).values
            )
            for ni, (nl, mf) in enumerate(zip(noise_levels, mean_frac)):
                if mf > 0.3:
                    x_lo = nl - (noise_levels[1] - noise_levels[0]) * 0.4 if ni > 0 else nl - 0.02
                    x_hi = nl + (noise_levels[min(ni + 1, len(noise_levels) - 1)] - nl) * 0.4
                    ax.axvspan(x_lo, x_hi, color="red", alpha=0.06, zorder=0)

        if ax_idx == 0:
            leg = ax.legend(title="Interp path", fontsize=7, title_fontsize=7)
            if _has_near_zero:
                # Add a manual entry for the near-zero overlay
                leg.get_frame().set_alpha(0.9)
                ax.plot([], [], color="dimgrey", linewidth=1.0, linestyle=":",
                        alpha=0.7, label="near-zero frac (right axis)")
                ax.legend(title="Interp path", fontsize=7, title_fontsize=7)

    # Hide unused panels
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    prot_label = protein_prefix or f"{protein}_{fitness_model}"
    split_label = f" [{split}]" if split and split != "all" else ""
    fig.suptitle(
        f"IG quality vs noise level — {protein} / {fitness_model}{split_label}\n"
        f"(lines = interpolation paths; dashed grey = random baseline; "
        f"dotted right axis = near-zero fraction; red shading = >30% unreliable)",
        fontsize=9,
    )
    fname = f"{prot_label}_noise_vs_interp.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_position_comparison(
    results_dir: Path,
    protein: str,
    gt: dict,
    ig_vec: np.ndarray,
    output_dir: Path,
    label: str = "",
) -> None:
    """Bar chart of |IG| vs ground-truth importance per position for one run."""
    L = gt["sequence_length"]
    positions = np.arange(1, L + 1)
    gt_vec = gt["importance_vector"]
    mask = gt["annotated_mask"]

    fig, axes = plt.subplots(2, 1, figsize=(max(14, L // 3), 6),
                             sharex=True, constrained_layout=True)

    # Top: IG scores coloured by annotation status
    colours = np.where(mask, "steelblue", "lightgrey")
    axes[0].bar(positions, ig_vec, color=colours, edgecolor="none", width=0.8)
    axes[0].set_ylabel("|IG| (mean abs gradient)", fontsize=8)
    axes[0].set_title(f"{label} — |IG| per position", fontsize=9)
    # Legend patches
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[Patch(color="steelblue", label="annotated"),
                 Patch(color="lightgrey", label="unannotated")],
        fontsize=7,
    )

    # Bottom: ground truth importance
    gt_colours = np.where(mask, "darkorange", "lightgrey")
    axes[1].bar(positions, gt_vec, color=gt_colours, edgecolor="none", width=0.8)
    axes[1].set_ylabel("Ground-truth importance", fontsize=8)
    axes[1].set_title("Ground-truth importance per position", fontsize=9)
    axes[1].set_xlabel("Position (1-indexed)", fontsize=8)
    axes[1].legend(
        handles=[Patch(color="darkorange", label="annotated"),
                 Patch(color="lightgrey", label="unannotated")],
        fontsize=7,
    )

    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    out_path = output_dir / f"{safe_label}_per_position.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic IG results against ground-truth position importance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dirs",
        nargs="*",
        help="Results directories to evaluate. If omitted, all synthetic results are "
             "discovered automatically under results/synthetic/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SYNTHETIC_RESULTS_DIR / "evaluation"),
        help="Where to write summary CSV and plots (default: results/synthetic/evaluation).",
    )
    parser.add_argument(
        "--use-per-variant",
        action="store_true",
        help="Use per-variant integrated_gradients_results.csv instead of the "
             "pre-aggregated position_importance_summary.csv.",
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover / validate results directories ──────────────────────────────
    if args.results_dirs:
        dirs = [Path(d) for d in args.results_dirs]
    else:
        print(f"Scanning for synthetic results under {SYNTHETIC_RESULTS_DIR} ...")
        all_dirs = discover_synthetic_results()
        # Apply optional protein / fitness-model filters
        def _matches(d: Path) -> bool:
            # Walk up to find the protein_model level (e.g. GB1_additive)
            parts = d.relative_to(SYNTHETIC_RESULTS_DIR).parts
            top = parts[0] if parts else ""
            if args.proteins and not any(top.startswith(p + "_") for p in args.proteins):
                return False
            if args.fitness_models and not any(
                f"_{fm}" in top for fm in args.fitness_models
            ):
                return False
            return True
        dirs = [d for d in all_dirs if _matches(d)]
        print(f"  Found {len(dirs)} result directories.\n")

    if not dirs:
        print("No results directories found. Exiting.")
        sys.exit(0)

    # ── Evaluate each directory ──────────────────────────────────────────────
    rows: list[dict] = []
    for d in dirs:
        print(f"Evaluating: {d.name}")

        # Detect whether this results dir has per-split tags in its IG CSV
        _ig_files = list(d.glob("*_integrated_gradients_results.csv"))
        _has_splits = False
        if _ig_files:
            try:
                _sample = pd.read_csv(_ig_files[0], usecols=["sequence_name"], nrows=500)
                _tags = _sample["sequence_name"].apply(_extract_split_tag).unique().tolist()
                _has_splits = any(t in _tags for t in ("trainval", "test"))
            except Exception:
                pass

        # Determine which split_filter values to run
        _split_filters: list[str | None] = [None]
        if _has_splits:
            _split_filters += ["trainval", "test"]

        for split_filter in _split_filters:
            row = evaluate_one(d, use_per_variant=args.use_per_variant, split_filter=split_filter)
            if row is None:
                continue
            rows.append(row)

            _split_label = split_filter or "all"
            # Optional per-position plot (only for 'all' or per-variant mode)
            if args.per_position_plots and (split_filter is None or args.use_per_variant):
                try:
                    protein = row["protein"]
                    gt = load_ground_truth(protein)
                    L = gt["sequence_length"]
                    if args.use_per_variant:
                        pos_df = _load_per_variant_gradients(d, split_filter=split_filter)
                    else:
                        pos_df = _load_position_importance(d)
                    if pos_df is not None:
                        ig_vec = np.zeros(L, dtype=float)
                        for _, r in pos_df.iterrows():
                            p0 = int(r["position"]) - 1
                            if 0 <= p0 < L:
                                ig_vec[p0] = float(r["mean_abs_gradient"])
                        # Build a label that includes protein, model, noise level, and interp
                        # so each combination gets its own file, e.g.:
                        #   GB1_additive_noise_0.1_linear_all_per_position.png
                        prot_model = row.get("protein", "") + "_" + row.get("fitness_model", "")
                        noise_part = d.parent.name if d.parent.name.startswith("noise_") else ""
                        per_pos_label = "_".join(filter(None, [prot_model, noise_part, d.name, _split_label]))
                        plot_per_position_comparison(
                            d, protein, gt, ig_vec, output_dir,
                            label=per_pos_label)
                except Exception as exc:
                    print(f"  [warn] Per-position plot failed: {exc}")

            print(
                f"  [{_split_label}]  "
                f"ρ(GT)={row['spearman_r']:+.3f}  τ(GT)={row['kendall_tau']:+.3f}  "
                f"AUROC(GT)={row['auroc']:.3f}  AUPRC(GT)={row['auprc']:.3f}  "
                f"ρ(fit)={row.get('fitness_spearman_r', float('nan')):+.3f}  "
                f"AUROC(fit)={row.get('fitness_auroc', float('nan')):.3f}  "
                f"completeness(5%)={row.get('completeness_frac_pass_5pct', float('nan')):.2f}  "
                f"near_zero={row.get('n_near_zero_sequences', 0)}"
            )

    if not rows:
        print("No valid results found.")
        sys.exit(0)

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(rows)
    # Drop internal columns before CSV export
    export_df = summary_df.drop(columns=[c for c in ["_cliff_rows", "_cliff_threshold"] if c in summary_df.columns])
    col_order = [
        "dirname", "protein", "fitness_model", "ig_method", "attribution", "interp",
        "noise_level", "split",
        "spearman_r", "spearman_p", "kendall_tau", "kendall_p",
        "auroc", "auprc", "n_positions", "n_annotated",
        "precision_at_k", "precision_at_k_random", "precision_at_k_K",
        # [2] Normalized rank metrics (|IG| / |ΔF|)
        "norm_spearman_r", "norm_kendall_tau", "norm_auroc", "norm_auprc",
        # [1] Completeness axiom stats
        "completeness_mean_rel_error", "completeness_median_rel_error",
        "completeness_frac_pass_1pct", "completeness_frac_pass_5pct", "completeness_frac_pass_10pct",
        "n_completeness_sequences",
        # [4] Near-zero output-difference (unreliable attributions)
        "n_near_zero_sequences",
        "fitness_spearman_r", "fitness_spearman_p",
        "fitness_kendall_tau", "fitness_kendall_p",
        "fitness_auroc", "fitness_auprc", "n_fitness_positions",
        "fitness_precision_at_k", "fitness_precision_at_k_random",
        "sign_acc_active", "sign_acc_wt", "sign_acc_overall",
        "n_active_mut", "n_wt_positions",
        "linear_spearman_r", "linear_pearson_r", "n_prop_obs",
        "epi_interacting_mean_absig", "epi_noninteracting_mean_absig",
        "epi_delta_absig", "n_epi_pairs",
    ]
    col_order = [c for c in col_order if c in export_df.columns]
    export_df = export_df[col_order]

    csv_path = output_dir / "synthetic_ig_validation_summary.csv"
    export_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSummary CSV: {csv_path}")

    # ── Print table ──────────────────────────────────────────────────────────
    hdr_cols = [
        ("Directory",    "dirname",             40, ""),
        ("ρ(GT)",        "spearman_r",           7, "+.3f"),
        ("τ(GT)",        "kendall_tau",          7, "+.3f"),
        ("AUROC(GT)",    "auroc",                8,  ".3f"),
        ("AUPRC(GT)",    "auprc",                8,  ".3f"),
        ("ρ(fit)",       "fitness_spearman_r",   7, "+.3f"),
        ("τ(fit)",       "fitness_kendall_tau",  7, "+.3f"),
        ("AUROC(fit)",   "fitness_auroc",        9,  ".3f"),
        ("AUPRC(fit)",   "fitness_auprc",        9,  ".3f"),
        ("SignAcc(act)", "sign_acc_active",      10, ".3f"),
        ("SignAcc(all)", "sign_acc_overall",     10, ".3f"),
        ("LinSp",        "linear_spearman_r",    6, "+.3f"),
    ]
    hdr_cols = [(h, col, w, fmt) for h, col, w, fmt in hdr_cols if col in export_df.columns]
    header = "".join(f"{h:>{w}}  " for h, col, w, fmt in hdr_cols)
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header.replace("Directory", f"{'Directory':<40}", 1))
    print(sep)
    for _, row in export_df.iterrows():
        line = f"{str(row['dirname']):<40}  "
        for h, col, w, fmt in hdr_cols[1:]:
            val = row.get(col, float("nan"))
            try:
                fval = float(val)
                cell = format(fval, fmt) if np.isfinite(fval) else "nan"
            except (TypeError, ValueError):
                cell = "nan"
            line += f"{cell:>{w}}  "
        print(line)
    print("=" * len(header))

    # ── Plots ────────────────────────────────────────────────────────────────
    proteins_present = list(export_df["protein"].unique()) if "protein" in export_df.columns else []
    splits_present   = sorted(export_df["split"].unique().tolist()) if "split" in export_df.columns else ["all"]

    def _safe_plots(df_sub: pd.DataFrame, prefix: str) -> None:
        """Generate the standard set of plots for a given (protein × split) subset."""
        if df_sub.empty:
            return
        try:
            plot_metric_comparison(df_sub, output_dir, protein_prefix=prefix)
        except Exception as exc:
            print(f"  [warn] Metric comparison [{prefix}] failed: {exc}")
        try:
            plot_fitness_metric_comparison(df_sub, output_dir, protein_prefix=prefix)
        except Exception as exc:
            print(f"  [warn] Fitness metric comparison [{prefix}] failed: {exc}")
        try:
            plot_precision_at_k(df_sub, output_dir, protein_prefix=prefix)
        except Exception as exc:
            print(f"  [warn] Weighted pos accuracy [{prefix}] failed: {exc}")
        try:
            plot_completeness_summary(df_sub, output_dir, protein_prefix=prefix)
        except Exception as exc:
            print(f"  [warn] Completeness summary [{prefix}] failed: {exc}")
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

    # Per-protein × per-split plots
    fitness_models_present = (
        list(export_df["fitness_model"].unique()) if "fitness_model" in export_df.columns else []
    )
    for prot in proteins_present:
        prot_df = export_df[export_df["protein"] == prot]
        for split in splits_present:
            split_df = prot_df[prot_df["split"] == split] if "split" in prot_df.columns else prot_df
            if split_df.empty:
                continue
            # Build prefix: e.g. "GB1_all", "GB1_trainval", "GB1_test"
            prefix = f"{prot}_{split}" if split != "all" else prot
            print(f"\n  Generating plots for protein={prot} split={split} ({len(split_df)} runs)")
            _safe_plots(split_df, prefix)

            # Per-(protein, fitness_model) noise-vs-interp line plots
            for fm in fitness_models_present:
                fm_df = split_df[split_df["fitness_model"] == fm] if "fitness_model" in split_df.columns else pd.DataFrame()
                if fm_df.empty or "noise_level" not in fm_df.columns or fm_df["noise_level"].isna().all():
                    continue
                fm_prefix = f"{prefix}_{fm}" if split != "all" else f"{prot}_{fm}"
                print(f"    Noise-vs-interp plot: {prot} / {fm} (split={split}, {len(fm_df)} runs)")
                try:
                    plot_noise_vs_interp(fm_df, output_dir, protein=prot, fitness_model=fm, protein_prefix=fm_prefix, split=split)
                except Exception as exc:
                    print(f"  [warn] Noise-vs-interp [{fm_prefix}] failed: {exc}")


    # Combined plots across all proteins — one set per split
    for split in splits_present:
        split_df = export_df[export_df["split"] == split] if "split" in export_df.columns else export_df
        if split_df.empty:
            continue
        combined_prefix = f"combined_{split}" if split != "all" else None
        print(f"\n  Generating combined overview plots (split={split}, {len(split_df)} runs)")
        _safe_plots(split_df, combined_prefix)

    print(f"\nDone. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
