"""
Attribution quality metrics for IGLOO.

Provides metric computation functions for evaluating integrated-gradient
attribution quality against ground-truth synthetic data.  All functions are
self-contained and do not depend on any IGLOO model or training code.

Public API
----------
- ``compute_metrics_for_dir(results_dir, protein, ...)`` → dict | None
- ``_agg_metrics(metrics_list)`` → (mean_dict, std_dict)
- ``discover_result_dirs(root, ...)`` → list of (protein, fm, noise, interp, path)
- ``discover_result_dirs_multiseed(root, ...)`` → list of (protein, fm, noise, seed, interp, path)
- ``_ndcg_1d(true_relevance, scores)`` → float
- ``_ap_graded(true_relevance, scores)`` → float
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score

# ---------------------------------------------------------------------------
# Project-layout helpers
# ---------------------------------------------------------------------------
_PKG_DIR      = Path(__file__).resolve().parent        # src/igloo/
_PROJECT_ROOT = _PKG_DIR.parent.parent                 # project root
SYNTHETIC_DATA_DIR = _PROJECT_ROOT / "data" / "synthetic"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.igloo.synthetic_data import PROTEINS as _PROTEIN_DEFS         # type: ignore
    from src.igloo.synthetic_data import blosum_penalty as _blosum_penalty  # type: ignore
    from src.igloo.synthetic_data import (                                   # type: ignore
        fitness_additive  as _fitness_additive,
        fitness_epistatic as _fitness_epistatic,
        fitness_nonlinear as _fitness_nonlinear,
    )
    _FIT_FNS: dict[str, Any] = {
        "additive":  _fitness_additive,
        "epistatic": _fitness_epistatic,
        "nonlinear": _fitness_nonlinear,
    }
except ImportError:
    _PROTEIN_DEFS = {}
    _FIT_FNS = {}

    def _blosum_penalty(wt_aa: str, mut_aa: str) -> float:  # type: ignore[misc]
        return 0.0 if wt_aa == mut_aa else 0.5

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKGROUND_IMPORTANCE = 0.05
NEAR_ZERO_THRESHOLD   = 0.01
_KNOWN_PROTEINS: list[str] = ["GB1", "T4Lysozyme", "p53"]

# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------
_GT_CACHE: dict[str, dict] = {}


def _load_gt_json(protein: str) -> dict:
    """Load (and cache) the ground-truth JSON for *protein*.

    Returns a dict with keys:
        importance_vector, annotated_mask, annotated_positions,
        annotations, category_weights, sequence_length.
    """
    if protein in _GT_CACHE:
        return _GT_CACHE[protein]
    gt_path = SYNTHETIC_DATA_DIR / f"{protein}_ground_truth_weights.json"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth JSON not found: {gt_path}. "
            f"Run SyntheticDatasetGenerator('{protein}').save_all() first."
        )
    with open(gt_path) as fh:
        data = json.load(fh)
    iv = np.array(data["importance_vector"], dtype=float)
    result = {
        "importance_vector":    iv,
        "annotated_mask":       iv > BACKGROUND_IMPORTANCE,
        "annotated_positions":  data.get("annotated_positions", {}),
        "annotations":          {int(k): v for k, v in data["annotations"].items()},
        "category_weights":     data.get("category_weights", {}),
        "sequence_length":      int(data["sequence_length"]),
    }
    _GT_CACHE[protein] = result
    return result


def _build_importance_vec(protein: str) -> tuple[np.ndarray, np.ndarray, dict, list[int]]:
    """Return (importance_vec, annotated_mask, by_cat, bg_positions) for *protein*."""
    gt  = _load_gt_json(protein)
    iv  = gt["importance_vector"]
    am  = gt["annotated_mask"]
    ann = gt["annotations"]  # int → category (0-indexed)

    by_cat: dict[str, list[int]] = {}
    for pos, cat in ann.items():
        by_cat.setdefault(cat, []).append(pos)

    bg_positions = [p for p in range(len(iv)) if p not in ann]
    return iv, am, by_cat, bg_positions


def _wt_sequence(protein: str) -> str:
    """Return the WT sequence for *protein* (empty string if unavailable)."""
    if protein in _PROTEIN_DEFS:
        return _PROTEIN_DEFS[protein].sequence
    return ""


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def discover_result_dirs(
    root: Path,
    proteins: list[str] | None = None,
    fitness_models: list[str] | None = None,
    name_contains: str | None = None,
) -> list[tuple[str, str, float, str, Path]]:
    """Walk *root* and return ``(protein, fitness_model, noise, interp, path)`` tuples.

    Expected hierarchy::

        root/{Protein}_{fitness_model}/noise_{level}/{interp}/
            *_integrated_gradients_results.csv

    Parameters
    ----------
    name_contains :
        If given, only top-level directories whose name contains this string
        are included (e.g. ``"deep"``).  If *None*, only directories whose
        name is exactly ``{Protein}_{fitness_model}`` are included (no suffix),
        preventing e.g. ``GB1_additive_deep`` from polluting a standard GB1 run.
    """
    _SUFFIXES = (
        "_integrated_gradients_results.csv",
        "_position_importance_summary.csv",
    )

    def _has_csv(d: Path) -> bool:
        try:
            return any(
                e.name.endswith(s) for e in d.iterdir() for s in _SUFFIXES if e.is_file()
            )
        except PermissionError:
            return False

    results: list[tuple[str, str, float, str, Path]] = []
    for level1 in sorted(root.iterdir()):
        if not level1.is_dir():
            continue
        prot_fm = level1.name
        protein = next((p for p in _KNOWN_PROTEINS if prot_fm.startswith(p)), None)
        fm      = next((f for f in ("additive", "epistatic", "nonlinear")
                        if f"_{f}" in prot_fm.lower()), None)
        if protein is None or fm is None:
            continue
        if proteins       and protein not in proteins:
            continue
        if fitness_models and fm      not in fitness_models:
            continue
        # Strict name matching: prevent e.g. GB1_additive_deep contaminating GB1_additive
        if name_contains is not None:
            if name_contains not in prot_fm:
                continue
        else:
            if prot_fm != f"{protein}_{fm}":
                continue
        found_noise = False
        for level2 in sorted(level1.iterdir()):
            if not level2.is_dir():
                continue
            noise_m = re.search(r"noise[_\-]([0-9]*\.?[0-9]+)", level2.name, re.I)
            if not noise_m:
                continue
            found_noise = True
            noise = float(noise_m.group(1))
            for level3 in sorted(level2.iterdir()):
                if not level3.is_dir() or not _has_csv(level3):
                    continue
                results.append((protein, fm, noise, level3.name.lower(), level3))
        # Flat layout: root/{Protein}_{fm}/{interp}/ with no noise subdirectory
        if not found_noise:
            for level2 in sorted(level1.iterdir()):
                if not level2.is_dir() or not _has_csv(level2):
                    continue
                results.append((protein, fm, 0.0, level2.name.lower(), level2))
    return results


def discover_result_dirs_multiseed(
    root: Path,
    proteins: list[str] | None = None,
    fitness_models: list[str] | None = None,
) -> list[tuple[str, str, float, str, str, Path]]:
    """Walk *root* and return ``(protein, fitness_model, noise, seed, interp, path)`` tuples.

    Supports two directory layouts::

        # New layout (no seed folder):
        root/{Protein}_{fitness_model}/noise_{level}/{interp}/
            *_integrated_gradients_results.csv

        # Legacy layout (with seed folder):
        root/{Protein}_{fitness_model}/noise_{level}/seed_{N}/{interp}/
            *_integrated_gradients_results.csv

    The ``seed`` field is set to ``"seed_42"`` for the new layout.
    """
    _SUFFIXES = (
        "_integrated_gradients_results.csv",
        "_position_importance_summary.csv",
    )

    def _has_csv(d: Path) -> bool:
        try:
            return any(
                e.name.endswith(s) for e in d.iterdir() for s in _SUFFIXES if e.is_file()
            )
        except PermissionError:
            return False

    results: list[tuple[str, str, float, str, str, Path]] = []
    for level1 in sorted(root.iterdir()):
        if not level1.is_dir():
            continue
        prot_fm = level1.name
        protein = next((p for p in _KNOWN_PROTEINS if prot_fm.startswith(p)), None)
        fm      = next((f for f in ("additive", "epistatic", "nonlinear")
                        if f"_{f}" in prot_fm.lower()), None)
        if protein is None or fm is None:
            continue
        if proteins       and protein not in proteins:
            continue
        if fitness_models and fm      not in fitness_models:
            continue
        for level2 in sorted(level1.iterdir()):
            if not level2.is_dir():
                continue
            noise_m = re.search(r"noise[_\-]([0-9]*\.?[0-9]+)", level2.name, re.I)
            if not noise_m:
                continue
            noise = float(noise_m.group(1))
            for level3 in sorted(level2.iterdir()):
                if not level3.is_dir():
                    continue
                if re.search(r"seed[_]?\d+", level3.name, re.I):
                    # Legacy layout: noise/seed/interp
                    seed = level3.name
                    for level4 in sorted(level3.iterdir()):
                        if not level4.is_dir() or not _has_csv(level4):
                            continue
                        results.append((protein, fm, noise, seed, level4.name.lower(), level4))
                elif _has_csv(level3):
                    # New layout: noise/interp (no seed folder)
                    results.append((protein, fm, noise, "seed_42", level3.name.lower(), level3))
    return results


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_ig_csv(results_dir: Path) -> pd.DataFrame | None:
    """Load ``*_integrated_gradients_results.csv`` from *results_dir*."""
    candidates = list(results_dir.glob("*_integrated_gradients_results.csv"))
    if not candidates:
        return None
    try:
        return pd.read_csv(candidates[0])
    except Exception:
        return None


def _is_wt(name: str) -> bool:
    s = str(name)
    return s == "WT" or s.lower().startswith("wt[")


# ---------------------------------------------------------------------------
# Scalar metric helpers
# ---------------------------------------------------------------------------

def _ndcg_1d(true_relevance: np.ndarray, scores: np.ndarray) -> float:
    """Full NDCG for a single (relevance, score) pair of 1-D arrays.

    *true_relevance* must be non-negative.  Background positions should be
    zeroed out before calling to prevent uninformative background-vs-background
    rank comparisons from contributing DCG variance.

    Returns NaN when fewer than 2 elements or all relevance is zero.
    """
    tr = np.asarray(true_relevance, dtype=float)
    sc = np.asarray(scores, dtype=float)
    if len(tr) < 2 or tr.sum() == 0:
        return float("nan")
    try:
        from sklearn.metrics import ndcg_score as _sk_ndcg
        return float(_sk_ndcg(tr.reshape(1, -1), sc.reshape(1, -1)))
    except Exception:
        return float("nan")


def _ap_graded(true_relevance: np.ndarray, scores: np.ndarray) -> float:
    """Weighted average precision at graded relevance (AP\@R).

    Extends classic average precision so each retrieved position contributes
    proportionally to its ground-truth importance weight::

        AP = (1/Z) * Σ_k  rel_k * (Σ_{j≤k} rel_j / k)

    Positions are sorted by descending *scores*.  *Z* = Σ rel_k.
    Background positions (relevance = 0) contribute nothing.

    Returns NaN when Z == 0 or fewer than 2 elements.
    """
    rel = np.asarray(true_relevance, dtype=float)
    sc  = np.asarray(scores, dtype=float)
    Z   = rel.sum()
    if Z == 0 or len(rel) < 2:
        return float("nan")
    order         = np.argsort(-sc)
    rel_sorted    = rel[order]
    cum_rel       = np.cumsum(rel_sorted)
    ranks         = np.arange(1, len(rel_sorted) + 1, dtype=float)
    weighted_prec = cum_rel / ranks
    return float(np.dot(rel_sorted, weighted_prec) / Z)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics_for_dir(
    results_dir: Path,
    protein: str,
    min_output_diff: float = NEAR_ZERO_THRESHOLD,
    min_count: int = 5,
    fitness_model: str = "additive",
    split_filter: str | None = None,
) -> dict[str, Any] | None:
    """Compute all attribution quality metrics for one results directory.

    Parameters
    ----------
    results_dir :
        Directory containing ``*_integrated_gradients_results.csv``.
    protein :
        Protein name (must have a ground-truth JSON in ``data/synthetic/``).
    min_output_diff :
        Sequences whose ``|effect_vs_baseline|`` is below this threshold are
        considered collapsed and excluded from position-level aggregation.
    min_count :
        Minimum number of sequences required to observe a position before
        including it in the reliability mask.
    fitness_model :
        One of ``'additive'``, ``'epistatic'``, ``'nonlinear'``.  Used to
        look up the LOO fitness function.
    split_filter :
        If ``'trainval'`` or ``'test'``, restrict to sequences whose name
        contains that tag.  ``None`` keeps all sequences.

    Returns
    -------
    dict | None
        Flat metrics dict, or ``None`` if no usable data is found.
    """
    ig_df = _load_ig_csv(results_dir)
    if ig_df is None or ig_df.empty:
        return None

    try:
        iv, am, by_cat, bg_positions = _build_importance_vec(protein)
    except (ValueError, FileNotFoundError):
        return None

    L      = len(iv)
    wt_seq = _wt_sequence(protein)

    df = ig_df[~ig_df["sequence_name"].apply(_is_wt)].copy()
    if df.empty:
        return None

    if split_filter is not None:
        df = df[df["sequence_name"].str.contains(f"[{split_filter}]", regex=False)].copy()
        if df.empty:
            return None

    # ── Collapse stats ────────────────────────────────────────────────────────
    seq_diffs = df.groupby("sequence_name")["effect_vs_baseline"].first()
    n_total   = len(seq_diffs)
    n_coll    = int((seq_diffs.abs() < min_output_diff).sum())
    coll_frac = n_coll / n_total if n_total > 0 else float("nan")

    valid_seqs = seq_diffs[seq_diffs.abs() >= min_output_diff].index
    df_stable  = df[df["sequence_name"].isin(valid_seqs)]
    if df_stable.empty:
        return None

    # ── Per-position mean |IG| / signed IG ───────────────────────────────────
    agg = (
        df_stable.groupby("position")["integrated_gradient"]
        .agg(mean_abs=lambda x: x.abs().mean(),
             mean_signed=lambda x: x.mean(),
             count="count")
        .reset_index()
    )
    ig_vec        = np.zeros(L, dtype=float)
    ig_vec_signed = np.zeros(L, dtype=float)
    count_vec     = np.zeros(L, dtype=int)
    for _, row in agg.iterrows():
        p0 = int(row["position"]) - 1
        if 0 <= p0 < L:
            ig_vec[p0]        = float(row["mean_abs"])
            ig_vec_signed[p0] = float(row["mean_signed"])
            count_vec[p0]     = int(row["count"])

    reliable   = count_vec >= min_count
    n_reliable = int(reliable.sum())
    ig_r       = ig_vec[reliable]
    ig_r_s     = ig_vec_signed[reliable]
    iv_r       = iv[reliable]
    am_r       = am[reliable]

    # ── Rank metrics helper ───────────────────────────────────────────────────
    def _rank_metrics(scores: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float]:
        """Return (spearman, auroc, auprc, prec_k, ndcg, ap_graded, pearson_r, mae_z, r2_z).

        ``pearson_r`` is a regression-style companion to ``spearman``: both
        compare ``scores`` (|IG| or signed IG, pooled per position) against
        ``iv_r`` (the ground-truth importance weight per position), but
        Pearson also asks whether attribution *magnitude* scales linearly
        with true importance, not just whether the relative order matches.
        Pearson is still scale/offset-invariant though (a dilated, shifted
        copy of ``iv_r`` scores r=1.0) -- it is NOT a true magnitude-agreement
        test. ``mae_z``/``r2_z`` are: z-score ``scores`` and ``iv_r``
        independently (they live on incommensurable scales, so a raw
        MSE/MAE would be meaningless), then take MAE and R² of ``scores`` as
        a *prediction* of ``iv_r`` -- R² here is against the y=x identity
        line (sklearn's standard prediction-vs-truth R²), not the best-fit
        regression line, so unlike Pearson r² it is sensitive to a
        magnitude/shape mismatch even when the two are perfectly linearly
        correlated at some other slope.
        """
        sp = (
            float(spearmanr(scores, iv_r)[0])
            if n_reliable >= 10 and iv_r.std() > 0 and scores.std() > 0
            else float("nan")
        )
        pr = (
            float(pearsonr(scores, iv_r)[0])
            if n_reliable >= 10 and iv_r.std() > 0 and scores.std() > 0
            else float("nan")
        )
        if n_reliable >= 10 and iv_r.std() > 0 and scores.std() > 0:
            z_scores = (scores - scores.mean()) / scores.std()
            z_iv     = (iv_r   - iv_r.mean())   / iv_r.std()
            mae_z = float(mean_absolute_error(z_iv, z_scores))
            r2_z  = float(r2_score(z_iv, z_scores))
        else:
            mae_z = r2_z = float("nan")
        if am_r.sum() > 0 and am_r.sum() < len(am_r):
            au = float(roc_auc_score(am_r.astype(int), scores))
            ap = float(average_precision_score(am_r.astype(int), scores))
        else:
            au = ap = float("nan")
        K = int(am_r.sum())
        pk = (
            float(am_r[np.argsort(-scores)[:K]].mean())
            if K > 0 and K < len(scores) else float("nan")
        )
        iv_r_ann = np.where(am_r, iv_r, 0.0)  # zero background for NDCG/AP@R
        nd = _ndcg_1d(iv_r_ann, scores)
        ag = _ap_graded(iv_r_ann, scores)
        return sp, au, ap, pk, nd, ag, pr, mae_z, r2_z

    (sp_r,  auroc,   auprc,   prec_k,   ndcg_r,   ap_graded,   pearson_r,   mae_z,   r2_z)   = _rank_metrics(ig_r)
    (sp_rs, auroc_s, auprc_s, prec_k_s, ndcg_rs,  ap_graded_s, pearson_r_s, mae_z_s, r2_z_s) = _rank_metrics(ig_r_s)

    # ── Category means ────────────────────────────────────────────────────────
    def _cat_means_for(vec: np.ndarray) -> tuple[dict[str, float], float]:
        cm: dict[str, float] = {}
        for cat, positions in by_cat.items():
            valid_pos = [p for p in positions if p < L]
            cm[cat] = float(np.mean([vec[p] for p in valid_pos])) if valid_pos else float("nan")
        cm["background"] = (
            float(np.mean([vec[p] for p in bg_positions if p < L]))
            if bg_positions else float("nan")
        )
        non_bg = {k: v for k, v in cm.items() if k != "background" and np.isfinite(v)}
        if non_bg and cm.get("background", 0) > 0:
            top = max(non_bg, key=non_bg.__getitem__)
            sep = cm[top] / cm["background"]
        else:
            sep = float("nan")
        return cm, sep

    cat_means,        sep        = _cat_means_for(ig_vec)
    cat_means_signed, sep_signed = _cat_means_for(ig_vec_signed)

    # ── Per-variant Spearman / NDCG ───────────────────────────────────────────
    pv_vals:              list[float] = []
    pv_vals_signed:       list[float] = []
    pv_ndcg_vals:         list[float] = []
    pv_ndcg_vals_signed:  list[float] = []
    pv_fitness_pairs:              list[tuple[float, float]] = []
    pv_fitness_pairs_signed:       list[tuple[float, float]] = []
    pv_ndcg_fitness_pairs:         list[tuple[float, float]] = []
    pv_ndcg_fitness_pairs_signed:  list[tuple[float, float]] = []

    iv_bg0 = np.where(am, iv, 0.0)  # background-zeroed relevance for per-variant NDCG

    for seq_name, seq_df in df_stable.groupby("sequence_name"):
        fitness_val = float(seq_df["effect_vs_baseline"].iloc[0])
        igv2        = np.zeros(L, dtype=float)
        igv2_s      = np.zeros(L, dtype=float)
        pos_arr     = seq_df["position"].values.astype(int) - 1
        ig_abs      = seq_df["integrated_gradient"].abs().values
        ig_raw      = seq_df["integrated_gradient"].values
        for idx, p in enumerate(pos_arr):
            if 0 <= p < L:
                igv2[p]   = ig_abs[idx]
                igv2_s[p] = ig_raw[idx]

        for vec_pv, out, fp_out in [
            (igv2,  pv_vals,        pv_fitness_pairs),
            (igv2_s, pv_vals_signed, pv_fitness_pairs_signed),
        ]:
            if vec_pv.std() == 0:
                out.append(0.0)
                fp_out.append((fitness_val, 0.0))
            else:
                r = float(spearmanr(vec_pv, iv)[0])
                if np.isfinite(r):
                    out.append(r)
                    fp_out.append((fitness_val, r))

        for _ndcg_vec, _ndcg_out, _ndcg_fp in [
            (igv2,  pv_ndcg_vals,        pv_ndcg_fitness_pairs),
            (igv2_s, pv_ndcg_vals_signed, pv_ndcg_fitness_pairs_signed),
        ]:
            _nd = _ndcg_1d(iv_bg0, _ndcg_vec)
            if np.isfinite(_nd):
                _ndcg_out.append(_nd)
                _ndcg_fp.append((fitness_val, _nd))

    def _nanmean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    def _nanstd(lst: list[float]) -> float:
        return float(np.std(lst)) if lst else float("nan")

    pv_mean             = _nanmean(pv_vals)
    pv_std              = _nanstd(pv_vals)
    pv_mean_signed      = _nanmean(pv_vals_signed)
    pv_std_signed       = _nanstd(pv_vals_signed)
    pv_ndcg_mean        = _nanmean(pv_ndcg_vals)
    pv_ndcg_std         = _nanstd(pv_ndcg_vals)
    pv_ndcg_mean_signed = _nanmean(pv_ndcg_vals_signed)
    pv_ndcg_std_signed  = _nanstd(pv_ndcg_vals_signed)

    def _vec_spearman(pairs: list[tuple[float, float]]) -> float:
        if len(pairs) < 5:
            return float("nan")
        fx = np.array([p[0] for p in pairs])
        fy = np.array([p[1] for p in pairs])
        if fx.std() == 0 or fy.std() == 0:
            return float("nan")
        return float(spearmanr(fx, fy)[0])

    def _vec_magnitude(pairs: list[tuple[float, float]]) -> tuple[float, float]:
        """Z-score-then-MAE/R² companion to ``_vec_spearman``, pooled across
        all sequences' mutations (not per-sequence -- too few points per
        sequence for a stable R²). fx = LOO fitness contribution (the
        ground-truth relevance), fy = |IG| or signed IG."""
        if len(pairs) < 5:
            return float("nan"), float("nan")
        fx = np.array([p[0] for p in pairs])
        fy = np.array([p[1] for p in pairs])
        if fx.std() == 0 or fy.std() == 0:
            return float("nan"), float("nan")
        zx = (fx - fx.mean()) / fx.std()
        zy = (fy - fy.mean()) / fy.std()
        return float(mean_absolute_error(zx, zy)), float(r2_score(zx, zy))

    pv_fitness_spearman        = _vec_spearman(pv_fitness_pairs)
    pv_fitness_spearman_signed = _vec_spearman(pv_fitness_pairs_signed)
    pv_fitness_ndcg            = _vec_spearman(pv_ndcg_fitness_pairs)
    pv_fitness_ndcg_signed     = _vec_spearman(pv_ndcg_fitness_pairs_signed)

    # ── Per-mutation contributions (LOO or additive fallback) ─────────────────
    _prot_def = _PROTEIN_DEFS.get(protein)
    _fit_fn   = _FIT_FNS.get(fitness_model)
    _use_loo  = _prot_def is not None and _fit_fn is not None and len(wt_seq) > 0
    if not _use_loo:
        _reason = (
            f"protein '{protein}' not in _PROTEIN_DEFS" if _prot_def is None
            else f"fitness_model '{fitness_model}' not in _FIT_FNS" if _fit_fn is None
            else "wt_seq unavailable"
        )
        warnings.warn(
            f"[ig_metrics] LOO unavailable ({_reason}); falling back to additive BLOSUM approx.",
            stacklevel=2,
        )

    mutation_pairs:            list[tuple[float, float]] = []
    mutation_pairs_signed:     list[tuple[float, float]] = []
    mutation_rho_vals:         list[float] = []
    mutation_rho_vals_signed:  list[float] = []
    mutation_pearson_vals:         list[float] = []
    mutation_pearson_vals_signed:  list[float] = []
    mutation_ndcg_vals:        list[float] = []
    mutation_ndcg_vals_signed: list[float] = []

    for _, seq_df in df_stable.groupby("sequence_name"):
        pos_arr = seq_df["position"].values.astype(int) - 1
        aa_arr  = seq_df["amino_acid"].values
        ig_abs  = seq_df["integrated_gradient"].abs().values
        ig_raw  = seq_df["integrated_gradient"].values

        mut_list: list[tuple[int, str, str]] = []
        mut_idxs: list[int] = []
        for idx, p in enumerate(pos_arr):
            if 0 <= p < L and p < len(wt_seq) and aa_arr[idx] != wt_seq[p]:
                mut_list.append((p, wt_seq[p], aa_arr[idx]))
                mut_idxs.append(idx)

        if _use_loo and len(mut_list) >= 2:
            _seq_arr = list(wt_seq)
            for p, _, ma in mut_list:
                _seq_arr[p] = ma
            _f_full = _fit_fn("".join(_seq_arr), _prot_def, mut_list)
            seq_contribs: list[float] = []
            for k, (p, wt_aa, mut_aa) in enumerate(mut_list):
                _rest = [m for i, m in enumerate(mut_list) if i != k]
                _seq_rest = list(wt_seq)
                for p2, _, ma2 in _rest:
                    _seq_rest[p2] = ma2
                _f_rest = _fit_fn("".join(_seq_rest), _prot_def, _rest)
                seq_contribs.append(float(_f_rest - _f_full))
        elif len(mut_list) >= 1:
            seq_contribs = [
                float(iv[p]) * _blosum_penalty(wt_aa, mut_aa)
                for p, wt_aa, mut_aa in mut_list
            ]
        else:
            continue

        seq_ig_abs = [ig_abs[i] for i in mut_idxs]
        seq_ig_raw = [ig_raw[i] for i in mut_idxs]
        for contrib, ia, ir in zip(seq_contribs, seq_ig_abs, seq_ig_raw):
            mutation_pairs.append((contrib, ia))
            mutation_pairs_signed.append((contrib, ir))

        if len(seq_contribs) >= 2:
            cx = np.array(seq_contribs)
            if cx.std() > 0:
                for ig_track, rho_list in [
                    (np.array(seq_ig_abs), mutation_rho_vals),
                    (np.array(seq_ig_raw), mutation_rho_vals_signed),
                ]:
                    r = float(spearmanr(cx, ig_track)[0])
                    if np.isfinite(r):
                        rho_list.append(r)
                # Pearson companion to the rank correlation above: does |IG|/
                # signed IG scale linearly with the LOO fitness delta, not
                # just agree in relative order.
                for ig_track, pearson_list in [
                    (np.array(seq_ig_abs), mutation_pearson_vals),
                    (np.array(seq_ig_raw), mutation_pearson_vals_signed),
                ]:
                    pr_val = float(pearsonr(cx, ig_track)[0])
                    if np.isfinite(pr_val):
                        pearson_list.append(pr_val)
            _cx_clip = np.clip(cx, 0, None)
            if _cx_clip.sum() > 0:
                for ig_track, ndcg_list in [
                    (np.array(seq_ig_abs), mutation_ndcg_vals),
                    (np.array(seq_ig_raw), mutation_ndcg_vals_signed),
                ]:
                    _nd = _ndcg_1d(_cx_clip, ig_track)
                    if np.isfinite(_nd):
                        ndcg_list.append(_nd)

    mutation_spearman        = _vec_spearman(mutation_pairs)
    mutation_spearman_signed = _vec_spearman(mutation_pairs_signed)
    mutation_mae_z,   mutation_r2_z   = _vec_magnitude(mutation_pairs)
    mutation_mae_z_s, mutation_r2_z_s = _vec_magnitude(mutation_pairs_signed)
    if mutation_pairs:
        _m_c  = np.array([p[0] for p in mutation_pairs])
        _m_ia = np.array([p[1] for p in mutation_pairs])
        _m_ir = np.array([p[1] for p in mutation_pairs_signed])
        _m_cc = np.clip(_m_c, 0, None)
        mutation_ndcg        = _ndcg_1d(_m_cc, _m_ia) if _m_cc.sum() > 0 else float("nan")
        mutation_ndcg_signed = _ndcg_1d(_m_cc, _m_ir) if _m_cc.sum() > 0 else float("nan")
    else:
        mutation_ndcg = mutation_ndcg_signed = float("nan")

    # ── Sign accuracy ─────────────────────────────────────────────────────────
    ann_set = set(_load_gt_json(protein)["annotations"].keys())
    corr = tot = 0
    for _, seq_df in df_stable.groupby("sequence_name"):
        thr     = float(seq_df["integrated_gradient"].abs().median())
        pos_arr = seq_df["position"].values.astype(int) - 1
        aa_arr  = seq_df["amino_acid"].values
        ig_arr  = seq_df["integrated_gradient"].values
        for idx, p in enumerate(pos_arr):
            if p in ann_set and p < len(wt_seq) and aa_arr[idx] != wt_seq[p]:
                tot += 1
                if ig_arr[idx] < -thr:
                    corr += 1
    sign_acc = corr / tot if tot > 0 else float("nan")

    # ── Output diff stats ─────────────────────────────────────────────────────
    out_diff_mean = float(seq_diffs.mean())
    out_diff_std  = float(seq_diffs.std())

    # ── Model predictive performance ──────────────────────────────────────────
    model_test_spearman = model_val_spearman  = float("nan")
    model_test_kendall  = float("nan")
    model_test_ndcg     = model_val_ndcg      = float("nan")
    model_test_r2       = model_val_r2        = float("nan")
    if "predicted_activity" in ig_df.columns and "actual_activity" in ig_df.columns:
        _pred_df = ig_df[~ig_df["sequence_name"].apply(_is_wt)].copy()
        _test_df = _pred_df[_pred_df["sequence_name"].str.contains("[test]", regex=False)]
        _seq_pred = _test_df.groupby("sequence_name")[
            ["predicted_activity", "actual_activity"]
        ].first()
        if len(_seq_pred) >= 5:
            _pa = _seq_pred["predicted_activity"].values
            _aa = _seq_pred["actual_activity"].values
            if _pa.std() > 0 and _aa.std() > 0:
                model_test_spearman = float(spearmanr(_pa, _aa)[0])
                # tau-b: same (pred, true) pairs as the Spearman above, but a
                # concordance-based rank correlation (tau = 2*P(concordant) - 1
                # for continuous data), so an x-axis value reads directly as
                # "how often the model orders a random variant pair correctly".
                model_test_kendall = float(kendalltau(_pa, _aa, variant="b")[0])
                _ss_res = np.sum((_aa - _pa) ** 2)
                _ss_tot = np.sum((_aa - _aa.mean()) ** 2)
                model_test_r2 = float(1 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
                _aa_rel = _aa - _aa.min()
                if _aa_rel.sum() > 0:
                    model_test_ndcg = _ndcg_1d(_aa_rel, _pa)
        _cv_path = next(results_dir.glob("*_cv_fold_metrics.csv"), None)
        if _cv_path is not None:
            try:
                _cv = pd.read_csv(_cv_path)
                if "val_spearman_rho" in _cv.columns:
                    model_val_spearman = float(_cv["val_spearman_rho"].mean())
                if "val_r2" in _cv.columns:
                    model_val_r2 = float(_cv["val_r2"].mean())
            except Exception:
                pass

    return {
        # Abs IG track
        "spearman":             sp_r,
        "pearson_r":            pearson_r,
        "mae_z":                mae_z,
        "r2_z":                 r2_z,
        "ndcg":                 ndcg_r,
        "auroc":                auroc,
        "auprc":                auprc,
        "prec_k":               prec_k,
        "ap_graded":            ap_graded,
        "pv_mean":              pv_mean,
        "pv_std":               pv_std,
        "pv_vals":              pv_vals,
        "pv_ndcg_mean":         pv_ndcg_mean,
        "pv_ndcg_std":          pv_ndcg_std,
        "pv_ndcg_vals":         pv_ndcg_vals,
        "cat_sep":              sep,
        "cat_means":            cat_means,
        "pv_fitness_spearman":  pv_fitness_spearman,
        "pv_fitness_ndcg":      pv_fitness_ndcg,
        "pv_fitness_pairs":     pv_fitness_pairs,
        "mutation_spearman":    mutation_spearman,
        "mutation_mae_z":       mutation_mae_z,
        "mutation_r2_z":        mutation_r2_z,
        "mutation_ndcg":        mutation_ndcg,
        "mutation_pairs":       mutation_pairs,
        "mutation_rho_vals":    mutation_rho_vals,
        "mutation_pearson_vals": mutation_pearson_vals,
        "mutation_ndcg_vals":   mutation_ndcg_vals,
        # Signed IG track
        "spearman_signed":              sp_rs,
        "pearson_r_signed":             pearson_r_s,
        "mae_z_signed":                 mae_z_s,
        "r2_z_signed":                  r2_z_s,
        "ndcg_signed":                  ndcg_rs,
        "auroc_signed":                 auroc_s,
        "auprc_signed":                 auprc_s,
        "prec_k_signed":                prec_k_s,
        "ap_graded_signed":             ap_graded_s,
        "pv_mean_signed":               pv_mean_signed,
        "pv_std_signed":                pv_std_signed,
        "pv_vals_signed":               pv_vals_signed,
        "pv_ndcg_mean_signed":          pv_ndcg_mean_signed,
        "pv_ndcg_std_signed":           pv_ndcg_std_signed,
        "pv_ndcg_vals_signed":          pv_ndcg_vals_signed,
        "cat_sep_signed":               sep_signed,
        "cat_means_signed":             cat_means_signed,
        "pv_fitness_spearman_signed":   pv_fitness_spearman_signed,
        "pv_fitness_ndcg_signed":       pv_fitness_ndcg_signed,
        "pv_fitness_pairs_signed":      pv_fitness_pairs_signed,
        "mutation_spearman_signed":     mutation_spearman_signed,
        "mutation_mae_z_signed":        mutation_mae_z_s,
        "mutation_r2_z_signed":         mutation_r2_z_s,
        "mutation_ndcg_signed":         mutation_ndcg_signed,
        "mutation_pairs_signed":        mutation_pairs_signed,
        "mutation_rho_vals_signed":     mutation_rho_vals_signed,
        "mutation_pearson_vals_signed": mutation_pearson_vals_signed,
        "mutation_ndcg_vals_signed":    mutation_ndcg_vals_signed,
        # Shared
        "sign_acc":             sign_acc,
        "collapse_frac":        coll_frac,
        "n_reliable":           n_reliable,
        "ig_vec":               ig_vec,
        "ig_vec_signed":        ig_vec_signed,
        "out_diff_mean":        out_diff_mean,
        "out_diff_std":         out_diff_std,
        # Model predictive performance
        "model_test_spearman":  model_test_spearman,
        "model_test_kendall":   model_test_kendall,
        "model_val_spearman":   model_val_spearman,
        "model_test_ndcg":      model_test_ndcg,
        "model_val_ndcg":       model_val_ndcg,
        "model_test_r2":        model_test_r2,
        "model_val_r2":         model_val_r2,
    }


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _agg_metrics(metrics_list: list[dict]) -> tuple[dict, dict]:
    """Aggregate a list of per-seed metric dicts into ``(mean_dict, std_dict)``.

    Non-scalar keys (lists, dicts) are silently dropped so that downstream
    ``dict.get(key, fallback)`` calls return their fallback correctly.
    """
    if not metrics_list:
        return {}, {}
    mean_d: dict = {}
    std_d:  dict = {}
    for k in metrics_list[0].keys():
        vals = [
            m[k] for m in metrics_list
            if isinstance(m.get(k), (int, float)) and np.isfinite(float(m[k]))
        ]
        if vals:
            mean_d[k] = float(np.mean(vals))
            std_d[k]  = float(np.std(vals, ddof=0))
    return mean_d, std_d
