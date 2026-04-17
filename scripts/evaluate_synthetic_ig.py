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

Usage
-----
    # Scan all synthetic results and compare
    python scripts/evaluate_synthetic_ig.py

    # Specific results directories
    python scripts/evaluate_synthetic_ig.py \\
        results/synthetic/GB1_additive_LINEAR \\
        results/synthetic/GB1_additive_SLERP

    # Control output location
    python scripts/evaluate_synthetic_ig.py --output-dir results/synthetic/evaluation

    # Use per-variant gradients (integrated_gradients_results.csv) instead of
    # the pre-aggregated position_importance_summary.csv
    python scripts/evaluate_synthetic_ig.py --use-per-variant
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


def _load_per_variant_gradients(results_dir: Path, dataset_name: str | None = None) -> pd.DataFrame | None:
    """Load the per-variant integrated_gradients_results.csv from *results_dir*.

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
    # Aggregate abs_gradient by position across all sequences
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


def _get_wt_sequence(ig_df: pd.DataFrame) -> str | None:
    """Reconstruct the wildtype sequence from the IG results dataframe.

    The IG result for the 'WT' sequence has amino_acid = wt residue at each position.
    Returns a 0-indexed string of length L, or None.
    """
    wt_rows = ig_df[ig_df["sequence_name"] == "WT"].sort_values("position")
    if wt_rows.empty:
        return None
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

    return {
        "spearman_r":  float(sp_r),
        "spearman_p":  float(sp_p),
        "kendall_tau": float(kt_tau),
        "kendall_p":   float(kt_p),
        "auroc":       float(auroc),
        "auprc":       float(auprc),
        "n_positions": int(len(ig_scores)),
        "n_annotated": int(y_true.sum()),
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
        if seq_name == "WT":
            continue
        for _, row in seq_df.iterrows():
            pos_0  = int(row["position"]) - 1
            ig_val = float(row["integrated_gradient"])
            mut_aa = str(row["amino_acid"])
            wt_aa  = wt_seq[pos_0] if pos_0 < len(wt_seq) else mut_aa

            is_mutated   = (mut_aa != wt_aa)
            is_annotated = bool(annotated_mask[pos_0]) if pos_0 < L else False

            if is_mutated:
                # Both annotated and unannotated mutations should produce negative IG
                correct = (ig_val < 0)
                all_correct += int(correct)
                all_total   += 1
                if is_annotated:
                    active_correct += int(correct)
                    active_total   += 1
            else:
                # WT residue → expect non-negative
                correct = (ig_val >= 0)
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
        if seq_name == "WT":
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
        if seq_name == "WT":
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
        if seq_name == "WT":
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
) -> None:
    """Bar chart of sign accuracy for each results directory.

    sign_rows : list of dicts, each with 'label', 'sign_acc_active',
                'sign_acc_wt', 'sign_acc_overall'.
    """
    if not sign_rows:
        return

    labels   = [r["label"] for r in sign_rows]
    acc_act  = [r["sign_acc_active"]  for r in sign_rows]
    acc_wt   = [r["sign_acc_wt"]      for r in sign_rows]
    acc_all  = [r["sign_acc_overall"] for r in sign_rows]

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5), constrained_layout=True)
    b1 = ax.bar(x - width, acc_act, width, label="Active-site (mutated+annotated)", color="steelblue",  alpha=0.85, edgecolor="k", linewidth=0.4)
    b2 = ax.bar(x,         acc_wt,  width, label="WT residues (unchanged)",         color="darkorange", alpha=0.85, edgecolor="k", linewidth=0.4)
    b3 = ax.bar(x + width, acc_all, width, label="Overall",                          color="seagreen",   alpha=0.85, edgecolor="k", linewidth=0.4)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.axhline(0.5, color="grey", lw=1.0, linestyle="--", label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Sign accuracy (fraction correct)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Sign Correctness: should mutations at critical sites produce negative IG?")
    ax.legend(fontsize=7, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    out_path = output_dir / "synthetic_ig_sign_accuracy.png"
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
) -> dict | None:
    """Evaluate a single results directory.

    Returns a flat dict of metadata + metrics, or None if data is missing.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"  [skip] Not a directory: {results_dir}")
        return None

    # ── Infer protein ────────────────────────────────────────────────────────
    protein = _parse_protein_from_dir(results_dir)
    if protein is None:
        print(f"  [skip] Cannot identify protein from dir name: {results_dir.name}")
        return None

    # ── Parse metadata from directory name ──────────────────────────────────
    # Expected patterns: GB1_additive_LINEAR  or  GB1_additive_embedding_signed
    dirname = results_dir.name
    fitness_model = next(
        (fm for fm in ["additive", "epistatic", "nonlinear"] if fm in dirname.lower()),
        "unknown",
    )
    ig_method = next(
        (m for m in ["embedding", "token"] if m in dirname.lower()),
        "unknown",
    )
    attribution = next(
        (a for a in ["signed", "magnitude"] if a in dirname.lower()),
        "unknown",
    )
    interp_match = re.search(
        r"(LINEAR|SLERP|NEAREST_NEIGHBOR|GUIDED_IG|LINEARINTERPOLATION)",
        dirname,
        re.IGNORECASE,
    )
    interp = interp_match.group(1).lower() if interp_match else "unknown"

    # ── Load ground truth ────────────────────────────────────────────────────
    try:
        gt = load_ground_truth(protein)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    L = gt["sequence_length"]

    # ── Load IG position scores ──────────────────────────────────────────────
    if use_per_variant:
        pos_df = _load_per_variant_gradients(results_dir)
        score_col = "mean_abs_gradient"
    else:
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

    # ── Detailed per-sequence IG for advanced metrics ────────────────────────
    ig_df = _load_detailed_ig(results_dir)
    wt_seq = _get_wt_sequence(ig_df) if ig_df is not None else None

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

    return {
        "results_dir":   str(results_dir),
        "dirname":       dirname,
        "protein":       protein,
        "fitness_model": fitness_model,
        "ig_method":     ig_method,
        "attribution":   attribution,
        "interp":        interp,
        **metrics,
        **sign_metrics,
        **prop_metrics,
        **epi_metrics,
        "_cliff_rows":      cliff_result["cliff_rows"],
        "_cliff_threshold": cliff_result["cliff_threshold"],
    }


# ---------------------------------------------------------------------------
# Discover results directories
# ---------------------------------------------------------------------------

def discover_synthetic_results(root: Path = SYNTHETIC_RESULTS_DIR) -> list[Path]:
    """Return all subdirectories under *root* that contain IG result CSVs."""
    dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        has_csv = (
            list(d.glob("*_position_importance_summary.csv"))
            or list(d.glob("*_integrated_gradients_results.csv"))
        )
        if has_csv:
            dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Grouped bar charts — one panel per metric, one bar per configuration."""
    metrics = ["spearman_r", "kendall_tau", "auroc", "auprc"]
    metric_labels = {
        "spearman_r":  "Spearman ρ",
        "kendall_tau": "Kendall τ",
        "auroc":       "AUROC",
        "auprc":       "AUPRC",
    }

    # Create a display label per row
    def _label(row):
        parts = [row["protein"], row["fitness_model"]]
        if row["ig_method"] != "unknown":
            parts.append(row["ig_method"])
        if row["attribution"] != "unknown":
            parts.append(row["attribution"])
        if row["interp"] not in ("unknown", ""):
            parts.append(row["interp"])
        return "\n".join(parts)

    summary_df = summary_df.copy()
    summary_df["label"] = summary_df.apply(_label, axis=1)

    n_configs = len(summary_df)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(max(10, n_configs * 0.7), 10),
        constrained_layout=True,
    )
    axes = axes.flatten()

    palette = plt.cm.tab20.colors
    colours = [palette[i % len(palette)] for i in range(n_configs)]

    for ax, metric in zip(axes, metrics):
        vals = summary_df[metric].values
        bars = ax.bar(
            range(n_configs), vals,
            color=colours, edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        # Reference line (AUROC/AUPRC random baseline)
        if metric in ("auroc",):
            ax.axhline(0.5, color="grey", lw=1.2, linestyle="--", label="random (0.5)")
            ax.legend(fontsize=7)
        # Annotate value above each bar
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(summary_df["label"].values, fontsize=6, ha="right", rotation=45)
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.set_title(metric_labels[metric], fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=min(0, float(np.nanmin(vals)) - 0.05))
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Synthetic IG Validation — Alignment with Ground-Truth Position Importance\n"
        "(Spearman/Kendall: rank correlation |IG| vs importance; "
        "AUROC/AUPRC: annotated vs unannotated positions)",
        fontsize=9,
    )

    out_path = output_dir / "synthetic_ig_validation_metrics.png"
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover / validate results directories ──────────────────────────────
    if args.results_dirs:
        dirs = [Path(d) for d in args.results_dirs]
    else:
        print(f"Scanning for synthetic results under {SYNTHETIC_RESULTS_DIR} ...")
        dirs = discover_synthetic_results()
        print(f"  Found {len(dirs)} result directories.\n")

    if not dirs:
        print("No results directories found. Exiting.")
        sys.exit(0)

    # ── Evaluate each directory ──────────────────────────────────────────────
    rows: list[dict] = []
    for d in dirs:
        print(f"Evaluating: {d.name}")
        row = evaluate_one(d, use_per_variant=args.use_per_variant)
        if row is None:
            continue
        rows.append(row)

        # Optional per-position plot
        if args.per_position_plots:
            try:
                protein = row["protein"]
                gt = load_ground_truth(protein)
                L = gt["sequence_length"]
                if args.use_per_variant:
                    pos_df = _load_per_variant_gradients(d)
                else:
                    pos_df = _load_position_importance(d)
                if pos_df is not None:
                    ig_vec = np.zeros(L, dtype=float)
                    for _, r in pos_df.iterrows():
                        p0 = int(r["position"]) - 1
                        if 0 <= p0 < L:
                            ig_vec[p0] = float(r["mean_abs_gradient"])
                    plot_per_position_comparison(d, protein, gt, ig_vec, output_dir,
                                                 label=d.name)
            except Exception as exc:
                print(f"  [warn] Per-position plot failed: {exc}")

        print(
            f"  ρ={row['spearman_r']:+.3f}  τ={row['kendall_tau']:+.3f}  "
            f"AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}"
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
        "spearman_r", "spearman_p", "kendall_tau", "kendall_p",
        "auroc", "auprc", "n_positions", "n_annotated",
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
        ("Directory", "dirname", 40, ""),
        ("ρ",         "spearman_r",  6, "+.3f"),
        ("τ",         "kendall_tau", 6, "+.3f"),
        ("AUROC",     "auroc",       6,  ".3f"),
        ("AUPRC",     "auprc",       6,  ".3f"),
        ("SignAcc(act)", "sign_acc_active",  10, ".3f"),
        ("SignAcc(all)", "sign_acc_overall", 10, ".3f"),
        ("LinSp",  "linear_spearman_r", 6, "+.3f"),
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
    try:
        plot_metric_comparison(export_df, output_dir)
    except Exception as exc:
        print(f"  [warn] Metric comparison plot failed: {exc}")

    # Sign accuracy bar chart
    try:
        sign_rows = [
            {"label": r["dirname"],
             "sign_acc_active":  r.get("sign_acc_active",  float("nan")),
             "sign_acc_wt":      r.get("sign_acc_wt",      float("nan")),
             "sign_acc_overall": r.get("sign_acc_overall", float("nan"))}
            for _, r in export_df.iterrows()
        ]
        plot_sign_accuracy_bars(sign_rows, output_dir)
    except Exception as exc:
        print(f"  [warn] Sign accuracy plot failed: {exc}")

    # Cliff damage plots (one per run that has cliff data)
    for row in rows:
        cliff_rows = row.get("_cliff_rows", [])
        if cliff_rows:
            try:
                # Attach threshold to each row for the plotting fn
                for cr in cliff_rows:
                    cr["cliff_threshold"] = row["_cliff_threshold"]
                plot_cliff_damage(cliff_rows, output_dir, label=row["dirname"])
            except Exception as exc:
                print(f"  [warn] Cliff plot for {row['dirname']} failed: {exc}")

    print(f"\nDone. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
