#!/usr/bin/env python3
"""
Embedding Distribution Analysis Script

Evaluates the quality of interpolation paths in embedding space using three analyses:

1. DATA DISTRIBUTION EXPLORATION
   - Histogram of pairwise cosine and euclidean distances (random subset)
   - Histograms of individual embedding dimensions (random sample of dims)
   - Histogram of per-sequence embedding norms

2. FRÉCHET EMBEDDING DISTANCE (FED)
   FED = ||μ_r - μ_int||² + Tr(Σ_r + Σ_int - 2(Σ_r Σ_int)^(1/2))
   Measures how in-distribution the interpolated path embeddings are
   relative to the real dataset embeddings.

3. ATTRIBUTE LINEAR SCORE (ALS)
   ALS = (1/N) Σ (y_actual(α_i) - y_ideal(α_i))²
   Measures how linearly the model's predicted activity changes along the
   interpolation path. (Ben notes this is likely not very meaningful since
   the assumption of linearity is probably wrong.)

Usage:
    # ── Single config ────────────────────────────────────────────────────────
    python scripts/analyze_embedding_distribution.py configs/pafa_config_embedding-signed.yaml
    python scripts/analyze_embedding_distribution.py configs/betalactamase_config_embedding-signed.yaml

    # Control how many sequence pairs to use for path analysis
    python scripts/analyze_embedding_distribution.py configs/pafa_config_embedding-signed.yaml --n-pairs 30

    # Run ALS (requires a trained model; skipped by default)
    python scripts/analyze_embedding_distribution.py configs/pafa_config_embedding-signed.yaml --run-als

    # ── Synthetic batch mode (RECOMMENDED for all synthetic datasets) ────────
    # Full production run over all proteins × fitness-models.
    # Requires: IG embedding cache files already generated in
    #   data/embeddings/synthetic/<dataset>_integrated_gradients_embeddings_batch_*.safetensors
    # Output goes to: results/synthetic/distribution_analysis/<dataset_name>/
    #
    source IGLOO_env/bin/activate
    python scripts/analyze_embedding_distribution.py \\
        --synthetic \\
        --n-pairs 20 \\
        --steps 500 \\
        --interp-methods linear slerp nearest_neighbor

    # Restrict to a single protein / fitness-model for testing:
    python scripts/analyze_embedding_distribution.py \\
        --synthetic \\
        --proteins T4Lysozyme \\
        --fitness-models additive \\
        --n-pairs 5 \\
        --steps 500 \\
        --interp-methods linear slerp nearest_neighbor

    # ── Pair-selection options (mutually exclusive intent) ───────────────────
    # Default: random pairs from non-WT sequences.
    #
    # Add top/bottom fitness extremes ON TOP of random pairs:
    #   --include-extremes --n-extremes 10
    #
    # Replace random selection with sequences farthest from embedding centroid:
    #   --embedding-extremes --n-embedding-extremes 20

    # ── Outputs per dataset ──────────────────────────────────────────────────
    # results/synthetic/distribution_analysis/<dataset>/
    #   <dataset>_individual_paths_linear.png
    #   <dataset>_individual_paths_slerp.png
    #   <dataset>_individual_paths_nearest_neighbor.png
    #   <dataset>_paths_method_comparison.png
    #   <dataset>_FED_boxplot.png
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import sqrtm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.igloo.config_utils import load_config, validate_config
from src.igloo.data_processing import (
    load_and_prepare_data,
    prepare_wildtype_sequence,
    set_global_normalization_params,
)
from src.igloo.embedding import get_embeddings
from src.igloo.sequence_ig import interpolate_embeddings, build_guided_ig_path


# ============================================================
# Utility helpers
# ============================================================

def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool (N, L, D) or (L, D) embeddings → (N, D) or (D,)."""
    if embeddings.ndim == 3:          # (N, L, D)
        return embeddings.mean(axis=1)
    elif embeddings.ndim == 2:        # (L, D) — single sequence
        return embeddings.mean(axis=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")


def compute_fed(real_embs: np.ndarray, path_embs: np.ndarray) -> float:
    """Compute Fréchet Embedding Distance between two sets of embeddings.

    Parameters
    ----------
    real_embs : (N1, D) – pooled embeddings from the real dataset.
    path_embs : (N2, D) – pooled embeddings sampled from interpolation paths.

    Returns
    -------
    Scalar FED value.
    """
    mu_r = real_embs.mean(axis=0)
    mu_p = path_embs.mean(axis=0)

    sigma_r = np.cov(real_embs, rowvar=False)
    sigma_p = np.cov(path_embs, rowvar=False)

    # Regularise to avoid singular matrices
    eps = 1e-6
    sigma_r += eps * np.eye(sigma_r.shape[0])
    sigma_p += eps * np.eye(sigma_p.shape[0])

    diff = mu_r - mu_p
    mean_term = float(diff @ diff) # sum of squared differences [DOUBLE CHECK THIS]
    mean_term_test = (np.sum((mu_r - mu_p) ** 2))

    # Matrix square root of (Σ_r Σ_p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sqrt_prod = sqrtm(sigma_r @ sigma_p)

    # Handle numerical imprecision — discard tiny imaginary parts
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real

    trace_term = float(np.trace(sigma_r + sigma_p - 2.0 * sqrt_prod))
    fed = mean_term + trace_term
    fed_test = mean_term_test + trace_term
    return fed_test


# ============================================================
# Step 1 — Data Distribution Exploration
# ============================================================

def plot_data_distribution(pooled_embs: np.ndarray, dataset_name: str,
                           output_dir: str, subset_size: int = 500) -> None:
    """
    Plot three diagnostic figures:
      (a) Pairwise cosine and euclidean distance histograms (random subset)
      (b) Per-dimension value histograms (random sample of 20 dimensions)
      (c) Per-sequence L2 norm histogram

    Parameters
    ----------
    pooled_embs : (N, D) mean-pooled embeddings.
    dataset_name : used for plot titles and filenames.
    output_dir : directory to save figures.
    subset_size : number of sequences used for pairwise distance calculation.
    """
    os.makedirs(output_dir, exist_ok=True)
    N, D = pooled_embs.shape
    print(f"\n[Step 1] Plotting data distribution for {N} sequences (dim={D})...")

    # ── (a) Pairwise distances ───────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=min(subset_size, N), replace=False)
    subset = pooled_embs[idx]

    # Pairwise euclidean distances (upper triangle only)
    diff = subset[:, None, :] - subset[None, :, :]          # (k, k, D)
    eucl_dists = np.sqrt((diff ** 2).sum(axis=-1))
    tri_idx = np.triu_indices(len(idx), k=1)
    eucl_vals = eucl_dists[tri_idx]

    # Pairwise cosine distances
    norms = np.linalg.norm(subset, axis=1, keepdims=True)
    normed = subset / np.clip(norms, 1e-8, None)
    cos_sim = normed @ normed.T
    cos_dists = 1.0 - cos_sim
    cos_vals = cos_dists[tri_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(eucl_vals, bins=60, color='steelblue', edgecolor='none', alpha=0.8)
    axes[0].set_title(f'{dataset_name}: Pairwise Euclidean Distance\n(n={len(idx)} sequences)')
    axes[0].set_xlabel('Euclidean Distance')
    axes[0].set_ylabel('Count')
    axes[0].axvline(float(np.median(eucl_vals)), color='red', lw=1.5, linestyle='--',
                    label=f'median={np.median(eucl_vals):.2f}')
    axes[0].legend()

    axes[1].hist(cos_vals, bins=60, color='darkorange', edgecolor='none', alpha=0.8)
    axes[1].set_title(f'{dataset_name}: Pairwise Cosine Distance\n(n={len(idx)} sequences)')
    axes[1].set_xlabel('Cosine Distance')
    axes[1].set_ylabel('Count')
    axes[1].axvline(float(np.median(cos_vals)), color='red', lw=1.5, linestyle='--',
                    label=f'median={np.median(cos_vals):.4f}')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_pairwise_distances.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── (b) Per-dimension distributions ─────────────────────────────────────
    n_dim_show = min(20, D)
    dim_idx = rng.choice(D, size=n_dim_show, replace=False)
    dim_idx.sort()

    ncols = 5
    nrows = int(np.ceil(n_dim_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    axes = axes.flatten()

    for i, d in enumerate(dim_idx):
        axes[i].hist(pooled_embs[:, d], bins=40, color='mediumseagreen', edgecolor='none', alpha=0.85)
        axes[i].set_title(f'dim {d}', fontsize=8)
        axes[i].tick_params(labelsize=6)
        axes[i].set_xlabel('value', fontsize=7)

    # Hide unused subplots
    for j in range(n_dim_show, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{dataset_name}: Sample of {n_dim_show} Embedding Dimensions', fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_dimension_histograms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── (c) Embedding norms ─────────────────────────────────────────────────
    norms_all = np.linalg.norm(pooled_embs, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(norms_all, bins=60, color='mediumpurple', edgecolor='none', alpha=0.85)
    ax.axvline(float(np.median(norms_all)), color='red', lw=1.5, linestyle='--',
               label=f'median={np.median(norms_all):.2f}')
    ax.axvline(float(np.mean(norms_all)), color='blue', lw=1.5, linestyle=':',
               label=f'mean={np.mean(norms_all):.2f}')
    ax.set_title(f'{dataset_name}: Per-Sequence Embedding Norms (N={N})')
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_embedding_norms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Print summary stats
    print(f"\n  Embedding stats (N={N}, D={D}):")
    print(f"    Euclidean distances: median={np.median(eucl_vals):.3f}, "
          f"mean={np.mean(eucl_vals):.3f}, std={np.std(eucl_vals):.3f}")
    print(f"    Cosine distances:    median={np.median(cos_vals):.4f}, "
          f"mean={np.mean(cos_vals):.4f}, std={np.std(cos_vals):.4f}")
    print(f"    Norms:               median={np.median(norms_all):.3f}, "
          f"mean={np.mean(norms_all):.3f}, std={np.std(norms_all):.3f}")


def plot_all_dimensions_kde(pooled_embs: np.ndarray, dataset_name: str,
                           output_dir: str,
                           seq_subsample: int = 200) -> None:
    """Plot a single figure with KDE curves for every embedding dimension
    overlaid on the same axes.

    Each curve is the density of values taken by one embedding dimension
    across all sequences. Overlaying all dimensions makes it easy to see
    whether they share a common distribution or whether some are outliers.

    Parameters
    ----------
    pooled_embs  : (N, D) mean-pooled embeddings.
    dataset_name : used for title and filename.
    output_dir   : directory to save the figure.
    seq_subsample: max number of sequences used for KDE estimation (speed).
    """
    from scipy.stats import gaussian_kde

    os.makedirs(output_dir, exist_ok=True)
    N, D = pooled_embs.shape
    print(f"\n[Step 1b] Plotting all-dimension KDE ({D} dims, N={N})...")

    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=min(seq_subsample, N), replace=False)
    data = pooled_embs[idx]   # (subsample, D)

    # Global x-range across all dims so curves are comparable
    global_min = float(data.min())
    global_max = float(data.max())
    x_grid = np.linspace(global_min, global_max, 400)

    # Compute per-dimension mean for colouring (low mean → blue, high mean → red)
    dim_means = data.mean(axis=0)   # (D,)
    norm_means = (dim_means - dim_means.min()) / (dim_means.ptp() + 1e-12)

    cmap = plt.cm.coolwarm

    fig, ax = plt.subplots(figsize=(11, 5))

    for d in range(D):
        col = data[:, d]
        # Skip nearly-constant dims (KDE would fail)
        if col.std() < 1e-8:
            continue
        kde = gaussian_kde(col, bw_method='silverman')
        density = kde(x_grid)
        ax.plot(x_grid, density,
                color=cmap(norm_means[d]),
                alpha=0.25, lw=0.6)

    # Overlay the grand mean KDE across all dims flattened
    flat = data.flatten()
    kde_all = gaussian_kde(flat, bw_method='silverman')
    ax.plot(x_grid, kde_all(x_grid),
            color='black', lw=2.5, zorder=5, label='All dims combined')

    ax.axvline(0, color='grey', lw=1, linestyle=':')
    ax.set_xlabel('Embedding Value')
    ax.set_ylabel('Density')
    ax.set_title(
        f'{dataset_name}: KDE of All {D} Embedding Dimensions\n'
        f'(N={len(idx)} sequences, colour = per-dim mean: blue=low, red=high)'
    )
    ax.legend(fontsize=9)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=float(dim_means.min()),
                                                   vmax=float(dim_means.max())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Per-dim mean value', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_all_dims_kde.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Step 2 — Fréchet Embedding Distance (FED)
# ============================================================

def collect_path_embeddings(sequences: list[str], wt_seq: str,
                             n_pairs: int, steps: int,
                             esm_layer: int, dataset_name: str,
                             interpolation_method: str = 'linear',
                             embeddings_subdir: str | None = None,
                             extra_seqs: list[str] | None = None,
                             predictor_model=None,
                             pooling_method: str | None = None,
                             y_log_mean: float = 0.0,
                             y_log_std: float = 1.0,
                             device: torch.device | None = None,
                             train_pooled_embs: np.ndarray | None = None,
                             precomputed_embs: np.ndarray | None = None,
                             ) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """
    Sample embeddings from interpolation paths.

    Returns
    -------
    all_path_pooled : (N_total, D) flat array of all path points (for FED).
    per_pair_paths  : list of (steps+1, D) arrays, one per pair.
    chosen_seqs     : list of variant sequences in the same order.
    """
    rng = np.random.default_rng(0)
    alphas = np.linspace(0.0, 1.0, steps + 1)

    # Pick random non-WT sequences
    non_wt = [s for s in sequences if s != wt_seq]
    chosen = rng.choice(len(non_wt), size=min(n_pairs, len(non_wt)), replace=False)
    chosen_seqs = [non_wt[i] for i in chosen]

    # Merge in extra (extreme) sequences, deduplicating
    if extra_seqs:
        existing = set(chosen_seqs)
        for s in extra_seqs:
            if s != wt_seq and s not in existing:
                chosen_seqs.append(s)
                existing.add(s)

    n_total = len(chosen_seqs)
    print(f"\n[Step 2] Collecting path embeddings ({n_total} pairs "
          f"[{min(n_pairs, len(non_wt))} random + {n_total - min(n_pairs, len(non_wt))} extremes] × "
          f"{steps} steps, method='{interpolation_method}')...")

    # Build a fast seq→row-index lookup from pre-fetched embeddings (shape N,L,D)
    # so we can avoid re-running ESM for sequences already in memory.
    _seq_to_idx: dict[str, int] = {}
    if precomputed_embs is not None:
        _seq_to_idx = {s: i for i, s in enumerate(sequences)}

    def _get_emb(seq: str, cache_name: str, bs: int) -> np.ndarray:
        """Return (1, L, D) raw embedding, using precomputed_embs when available."""
        if precomputed_embs is not None and seq in _seq_to_idx:
            return precomputed_embs[_seq_to_idx[seq]:_seq_to_idx[seq]+1]
        return get_embeddings([seq], layer=esm_layer, batch_size=bs,
                              dataset_name=cache_name, subdir=embeddings_subdir)

    # Pre-fetch WT embedding once
    wt_emb_raw = _get_emb(wt_seq, f'{dataset_name}_wt_baseline', 1)  # (1, L, D)
    wt_emb = torch.tensor(wt_emb_raw[0], dtype=torch.float32)   # (L, D)

    # Pre-fetch all chosen variant embeddings — use precomputed where available,
    # fall back to ESM only for sequences not already in memory.
    missing_seqs  = [s for s in chosen_seqs if s not in _seq_to_idx or precomputed_embs is None]
    missing_idxs  = [i for i, s in enumerate(chosen_seqs) if s not in _seq_to_idx or precomputed_embs is None]
    if missing_seqs:
        print(f"  Fetching {len(missing_seqs)} variant embedding(s) not in precomputed cache...")
        missing_embs = get_embeddings(
            missing_seqs, layer=esm_layer, batch_size=16,
            dataset_name=f'{dataset_name}_path_seqs',
            subdir=embeddings_subdir,
        )  # (M, L, D)
    else:
        missing_embs = None

    def _get_variant_emb(idx: int, seq: str) -> np.ndarray:
        if precomputed_embs is not None and seq in _seq_to_idx:
            return precomputed_embs[_seq_to_idx[seq]:_seq_to_idx[seq]+1]
        pos = missing_idxs.index(idx)
        return missing_embs[pos:pos+1]

    all_path_pooled: list[np.ndarray] = []
    per_pair_paths:  list[np.ndarray] = []

    for idx, seq in enumerate(tqdm(chosen_seqs, desc='  Pairs')):
        seq_emb_raw = _get_variant_emb(idx, seq)  # (1, L_seq, D)
        seq_emb = torch.tensor(seq_emb_raw[0], dtype=torch.float32)  # (L_seq, D)

        # Sequences may differ in length — pad shorter to match longer
        L_wt, L_s = wt_emb.shape[0], seq_emb.shape[0]
        if L_wt != L_s:
            L_max = max(L_wt, L_s)
            if L_wt < L_max:
                pad = torch.zeros(L_max - L_wt, wt_emb.shape[1])
                wt_emb_pad = torch.cat([wt_emb, pad], dim=0)
            else:
                wt_emb_pad = wt_emb
            if L_s < L_max:
                pad = torch.zeros(L_max - L_s, seq_emb.shape[1])
                seq_emb_pad = torch.cat([seq_emb, pad], dim=0)
            else:
                seq_emb_pad = seq_emb
        else:
            wt_emb_pad = wt_emb
            seq_emb_pad = seq_emb

        pair_points: list[np.ndarray] = []

        if interpolation_method == 'guided_ig':
            # build_guided_ig_path expects (1, L, D) tensors and a trained model
            _device = device or torch.device('cpu')
            path_tensors = build_guided_ig_path(
                wt_emb_pad.unsqueeze(0), seq_emb_pad.unsqueeze(0),
                predictor_model, pooling_method,
                y_log_mean, y_log_std, _device, steps,
            )  # list of (1, L, D), length = steps+1
            for k, x in enumerate(path_tensors):
                pooled = x[0].mean(dim=0).cpu().numpy()   # (D,)
                pair_points.append(pooled)
                # Exclude endpoints (idx 0 = WT, idx -1 = variant)
                if 0 < k < steps:
                    all_path_pooled.append(pooled)
        elif interpolation_method == 'nearest_neighbor':
            # Training-data-grounded path: at each α find the closest real
            # training embedding (in mean-pooled space) to the linearly
            # interpolated reference, mirroring _compute_path_integral.
            if train_pooled_embs is None:
                raise ValueError(
                    "interpolation_method='nearest_neighbor' requires "
                    "train_pooled_embs to be passed to collect_path_embeddings."
                )
            train_pt = torch.tensor(train_pooled_embs, dtype=torch.float32)  # (N, D)
            wt_pooled   = wt_emb_pad.mean(dim=0)    # (D,)
            seq_pooled  = seq_emb_pad.mean(dim=0)   # (D,)

            # γ_0 = baseline endpoint
            pair_points.append(wt_pooled.numpy())
            for alpha in alphas[1:-1]:
                ref_pooled = wt_pooled + float(alpha) * (seq_pooled - wt_pooled)  # (D,)
                dists  = torch.norm(train_pt - ref_pooled.unsqueeze(0), dim=-1)  # (N,)
                nn_idx = int(dists.argmin().item())
                nn_pooled = train_pt[nn_idx].numpy()  # (D,)
                pair_points.append(nn_pooled)
                all_path_pooled.append(nn_pooled)
            # γ_K = sequence endpoint
            pair_points.append(seq_pooled.numpy())
        else:  # linear, slerp
            for alpha in alphas:
                interp = interpolate_embeddings(wt_emb_pad, seq_emb_pad,
                                                float(alpha), method=interpolation_method)
                pooled = interp.mean(dim=0).numpy()   # (D,)
                pair_points.append(pooled)
                # Exclude endpoints (α=0 = WT, α=1 = variant) from the aggregate
                # path cloud — they are real sequences whose FED contribution is
                # trivially zero and would artificially lower the aggregate FED.
                if alpha not in (0.0, 1.0):
                    all_path_pooled.append(pooled)

        per_pair_paths.append(np.stack(pair_points, axis=0))  # (steps+1, D)

    return np.stack(all_path_pooled, axis=0), per_pair_paths, chosen_seqs


def compute_and_plot_fed(real_pooled: np.ndarray, path_pooled: np.ndarray,
                         dataset_name: str, output_dir: str,
                         interpolation_method: str = 'linear') -> float:
    """Compute FED and save a summary plot comparing the two distributions."""
    os.makedirs(output_dir, exist_ok=True)

    fed = compute_fed(real_pooled, path_pooled)
    print(f"\n  FED ({interpolation_method}): {fed:.4f}")

    # ── Compare norms distributions ─────────────────────────────────────────
    real_norms = np.linalg.norm(real_pooled, axis=1)
    path_norms = np.linalg.norm(path_pooled, axis=1)

    # Project onto first 2 principal components for visual check
    from sklearn.decomposition import PCA
    combined = np.concatenate([real_pooled, path_pooled], axis=0)
    combined_f64 = combined.astype(np.float64)
    pca = PCA(n_components=2, svd_solver='full')
    proj = pca.fit_transform(combined_f64)
    n_real = len(real_pooled)
    proj_real = proj[:n_real]
    proj_path = proj[n_real:]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Norm comparison
    axes[0].hist(real_norms, bins=50, alpha=0.6, label='Real', color='steelblue', density=True)
    axes[0].hist(path_norms, bins=50, alpha=0.6, label='Path', color='tomato', density=True)
    axes[0].set_title(f'{dataset_name}: Embedding Norm Distributions')
    axes[0].set_xlabel('L2 Norm')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # PCA scatter
    axes[1].scatter(proj_real[:, 0], proj_real[:, 1], s=4, alpha=0.4,
                    label=f'Real (n={n_real})', color='steelblue')
    axes[1].scatter(proj_path[:, 0], proj_path[:, 1], s=4, alpha=0.4,
                    label=f'Path (n={len(path_pooled)})', color='tomato')
    axes[1].set_title(f'{dataset_name}: PCA of Real vs Path Embeddings\n'
                      f'FED ({interpolation_method}) = {fed:.4f}')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(markerscale=3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_FED_{interpolation_method}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Per-dim mean ± std — all dimensions, separate wide figure ───────────
    D = real_pooled.shape[1]
    dims = np.arange(D)

    real_mean = real_pooled.mean(axis=0)
    real_std  = real_pooled.std(axis=0)
    path_mean = path_pooled.mean(axis=0)
    path_std  = path_pooled.std(axis=0)

    # Compute absolute difference between real and path means per dim
    mean_diff = np.abs(real_mean - path_mean)

    fig, axes_d = plt.subplots(2, 1, figsize=(max(16, D // 30), 8),
                               gridspec_kw={'height_ratios': [3, 1]})

    # Top panel: mean ± std
    axes_d[0].plot(dims, real_mean, color='steelblue', lw=1.0, label='Real mean')
    axes_d[0].fill_between(dims, real_mean - real_std, real_mean + real_std,
                            alpha=0.2, color='steelblue', label='Real ±1 std')
    axes_d[0].plot(dims, path_mean, color='tomato', lw=1.0, label='Path mean')
    axes_d[0].fill_between(dims, path_mean - path_std, path_mean + path_std,
                            alpha=0.2, color='tomato', label='Path ±1 std')
    axes_d[0].set_ylabel('Value')
    axes_d[0].set_title(
        f'{dataset_name}: Per-Dim Mean ± Std — All {D} Dimensions '
        f'({interpolation_method} path)'
    )
    axes_d[0].legend(fontsize=8)
    axes_d[0].set_xlim(0, D - 1)

    # Bottom panel: |real_mean - path_mean| per dim
    axes_d[1].bar(dims, mean_diff, width=1.0, color='purple', alpha=0.6)
    axes_d[1].set_ylabel('|Δ mean|')
    axes_d[1].set_xlabel('Embedding Dimension')
    axes_d[1].set_title('Absolute mean difference (Real − Path) per dimension')
    axes_d[1].set_xlim(0, D - 1)

    plt.tight_layout()
    perdim_path = os.path.join(output_dir,
                               f'{dataset_name}_perdim_mean_std_{interpolation_method}.png')
    plt.savefig(perdim_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {perdim_path}")

    return fed


def compute_per_pair_feds(
    real_pooled: np.ndarray,
    per_pair_paths: list[np.ndarray],
) -> list[float]:
    """Compute one FED value per interpolation path vs the real distribution.

    FED is computed directly in the full embedding space.  The path covariance
    is rank-deficient (51 points in 1280-D), but compute_fed already adds an
    eps * I regularisation term so the matrix square root remains stable.

    Parameters
    ----------
    real_pooled     : (N, D) mean-pooled real embeddings.
    per_pair_paths  : list of (steps+1, D) path arrays, one per variant.

    Returns
    -------
    List of FED floats, one per path in per_pair_paths.
    """
    feds: list[float] = []
    for path in per_pair_paths:
        # Exclude endpoints (α=0 = WT, α=1 = variant): they are real sequences
        # whose embeddings trivially sit inside the real distribution and would
        # artificially deflate the FED for every method equally.
        feds.append(compute_fed(real_pooled, path[1:-1]))
    return feds


def plot_fed_boxplot(
    fed_by_method: dict[str, list[float]],
    dataset_name: str,
    output_dir: str,
    chosen_seqs: list[str] | None = None,
) -> None:
    """Box plot comparing per-variant FED distributions across interpolation methods.

    One box per method; each data point is the FED for one variant path vs the
    real embedding distribution (computed in PCA-reduced space).

    Parameters
    ----------
    fed_by_method : dict mapping method name → list of per-variant FED values.
    dataset_name  : used for title and filename.
    output_dir    : directory to save the figure.
    chosen_seqs   : optional list of variant sequences (used to label outliers).
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = list(fed_by_method.keys())
    data    = [fed_by_method[m] for m in methods]

    # Colour palette — one colour per method
    palette = ['steelblue', 'darkorange', 'mediumseagreen',
               'mediumpurple', 'tomato', 'goldenrod']
    colours = palette[:len(methods)]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 2.5), 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color='black', lw=2),
        flierprops=dict(marker='o', markersize=4, linestyle='none',
                        markeredgecolor='black', alpha=0.7),
    )
    for patch, colour in zip(bp['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    for whisker, cap in zip(
        zip(bp['whiskers'][::2], bp['whiskers'][1::2]),
        zip(bp['caps'][::2],     bp['caps'][1::2]),
    ):
        for line in whisker + cap:
            line.set_color('dimgrey')

    # Overlay individual data points (jittered)
    rng = np.random.default_rng(42)
    for i, (vals, colour) in enumerate(zip(data, colours), start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, s=18, color=colour,
                   edgecolors='black', linewidths=0.4, alpha=0.85, zorder=3)

    # Annotate mean ± std per method, placed at the vertical midpoint of the plot
    all_vals_flat = [v for vs in data for v in vs]
    y_mid = (max(all_vals_flat) + min(all_vals_flat)) / 2 if all_vals_flat else 0.5
    for i, vals in enumerate(data, start=1):
        mean_val = float(np.mean(vals))
        std_val  = float(np.std(vals))
        ax.text(i, y_mid,
                f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                ha='center', va='center', fontsize=9, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=10)
    ax.set_ylabel('FED (per-variant path vs real distribution)', fontsize=9)
    ax.set_title(
        f'{dataset_name}: Per-Variant FED by Interpolation Method\n'
        f'(lower = path stays closer to real embedding distribution; '
        f'n={len(data[0])} variants per method)',
        fontsize=9,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{dataset_name}_FED_boxplot.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_individual_paths_pca(
    real_pooled: np.ndarray,
    per_pair_paths: list[np.ndarray],
    chosen_seqs: list[str],
    all_sequences: list[str],
    y_raw: np.ndarray,
    wt_seq: str,
    dataset_name: str,
    output_dir: str,
    interpolation_method: str = 'linear',
    ncols: int = 5,
    emb_extreme_set: set[str] | None = None,
    fit_top_set: set[str] | None = None,
    fit_bot_set: set[str] | None = None,
    per_pair_feds: list[float] | None = None,
) -> None:
    """One subplot per WT → variant path, projected into the PCA space of the
    real dataset.  Each subplot shows:
      • All real embeddings as a light grey background cloud
      • The WT embedding as a green star
      • The variant endpoint as a coloured dot (red=low fitness, blue=high)
      • The interpolation path as an orange gradient line WT → variant

    Subplot border colours indicate selection origin:
      Gold   = selected as embedding extreme (farthest from centroid)
      Blue   = selected as top-fitness extreme
      Red    = selected as bottom-fitness extreme
      Magenta= selected as BOTH embedding & fitness extreme
    """
    from sklearn.decomposition import PCA
    from matplotlib.collections import LineCollection

    os.makedirs(output_dir, exist_ok=True)
    n_pairs = len(per_pair_paths)
    print(f"\n  Plotting {n_pairs} individual paths in PCA space...")

    # Build seq → fitness lookup
    seq_to_y = {s: float(v) for s, v in zip(all_sequences, y_raw)}
    wt_y = seq_to_y.get(wt_seq, float('nan'))

    # Fit PCA on real data only, then project everything into that space
    pca = PCA(n_components=2, svd_solver='full')
    real_proj = pca.fit_transform(real_pooled.astype(np.float64))          # (N, 2)
    exp_var = pca.explained_variance_ratio_

    # WT projection
    if wt_seq in all_sequences:
        wt_proj = real_proj[all_sequences.index(wt_seq)]
    else:
        wt_proj = pca.transform(per_pair_paths[0][[0]])[0]

    # Project each path
    pair_projs = [pca.transform(path) for path in per_pair_paths]   # list of (S+1, 2)

    # Fitness colour scale
    all_y_vals = np.array([seq_to_y.get(s, float('nan')) for s in chosen_seqs])
    y_finite = all_y_vals[np.isfinite(all_y_vals)]
    y_min = float(y_finite.min()) if len(y_finite) else 0.0
    y_max = float(y_finite.max()) if len(y_finite) else 1.0
    fitness_cmap = plt.cm.RdBu

    # Grid layout
    nrows = int(np.ceil(n_pairs / ncols))
    cell = 2.8
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * cell, nrows * cell),
                              squeeze=False)

    # Consistent axis limits from real data
    margin = 0.05
    xr = real_proj[:, 0].ptp()
    yr = real_proj[:, 1].ptp()
    xlim = (real_proj[:, 0].min() - margin * xr,
            real_proj[:, 0].max() + margin * xr)
    ylim = (real_proj[:, 1].min() - margin * yr,
            real_proj[:, 1].max() + margin * yr)

    for i, (path_proj, seq) in enumerate(zip(pair_projs, chosen_seqs)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        # Background cloud
        ax.scatter(real_proj[:, 0], real_proj[:, 1],
                   s=2, color='lightgrey', alpha=0.5, zorder=1, rasterized=True)

        # Path as gradient line (faint near WT, opaque near variant)
        pts = path_proj[:, np.newaxis, :]                               # (S+1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)             # (S, 2, 2)
        seg_alphas = np.linspace(0.15, 1.0, len(segs))
        for seg, a in zip(segs, seg_alphas):
            ax.plot(seg[:, 0], seg[:, 1], color='darkorange',
                    lw=1.4, alpha=a, zorder=3, solid_capstyle='round')

        # Arrow on last segment
        ax.annotate('', xy=path_proj[-1], xytext=path_proj[-2],
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.4),
                    zorder=4)

        # WT star
        ax.scatter(*wt_proj, marker='*', s=130, color='limegreen',
                   edgecolors='black', linewidths=0.5, zorder=5)

        # Variant endpoint dot coloured by fitness
        var_y = seq_to_y.get(seq, float('nan'))
        norm_y = np.clip((var_y - y_min) / (y_max - y_min + 1e-12), 0, 1)
        ax.scatter(*path_proj[-1], s=60,
                   color=fitness_cmap(norm_y),
                   edgecolors='black', linewidths=0.5, zorder=6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        # Determine selection tags & border colour for this sequence
        is_emb = (emb_extreme_set is not None and seq in emb_extreme_set)
        is_top = (fit_top_set is not None and seq in fit_top_set)
        is_bot = (fit_bot_set is not None and seq in fit_bot_set)
        is_fit = is_top or is_bot
        tags = []
        if is_emb:
            tags.append('[E]')
        if is_top:
            tags.append('[↑]')
        if is_bot:
            tags.append('[↓]')
        tag_str = ' '.join(tags)

        if is_emb and is_fit:
            border_color, lw = 'magenta', 2.0
        elif is_emb:
            border_color, lw = 'goldenrod', 2.0
        elif is_top:
            border_color, lw = 'steelblue', 2.0
        elif is_bot:
            border_color, lw = 'tomato', 2.0
        else:
            border_color, lw = 'lightgrey', 0.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(lw)

        y_label = f'{var_y:.3f}' if np.isfinite(var_y) else '?'
        title = f'fitness={y_label}'
        if tag_str:
            title += f'  {tag_str}'
        if per_pair_feds is not None and i < len(per_pair_feds):
            title += f'\nFED={per_pair_feds[i]:.4f}'
        ax.set_title(title, fontsize=7, pad=2)

    # Hide unused axes
    for j in range(n_pairs, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    # Shared axis labels
    fig.text(0.5, 0.01, f'PC1 ({exp_var[0]*100:.1f}%)', ha='center', fontsize=9)
    fig.text(0.01, 0.5, f'PC2 ({exp_var[1]*100:.1f}%)', va='center',
             rotation='vertical', fontsize=9)

    plt.tight_layout()

    # Reserve a fixed strip on the right for the colourbar, then add it there
    # so it never overlaps with the subplot grid.
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.65])   # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=fitness_cmap,
                                norm=plt.Normalize(vmin=y_min, vmax=y_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Variant fitness', fontsize=8, labelpad=6)

    border_legend = (
        'Border: gold=[E] emb-extreme  |  blue=[↑] fit-top  |  '
        'red=[↓] fit-bot  |  magenta=both'
    )
    fig.suptitle(
        f'{dataset_name}: Individual WT → Variant Paths  ({interpolation_method})\n'
        f'Green ★ = WT (fitness={wt_y:.3f})  |  dot = variant (red=low, blue=high)  '
        f'|  orange = path\n{border_legend}',
        fontsize=8, y=1.01,
    )

    out_path = os.path.join(output_dir,
                            f'{dataset_name}_individual_paths_{interpolation_method}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Step 3 — Attribute Linear Score (ALS)
# ============================================================

def compute_als(sequences: list[str], y_raw: np.ndarray, wt_seq: str,
                n_pairs: int, steps: int, esm_layer: int, dataset_name: str,
                predictor_model, pooling_method: str, y_log_mean: float,
                y_log_std: float, device: torch.device,
                interpolation_method: str = 'linear',
                embeddings_subdir: str | None = None) -> dict:
    """
    Compute the Attribute Linear Score along interpolation paths.

    ALS = (1/N) Σ (y_actual(α_i) - y_ideal(α_i))²

    where:
      y_actual(α_i) = model prediction at the interpolated embedding at step α_i
      y_ideal(α_i)  = y_baseline + α_i * (y_sequence - y_baseline)  [linear interp]

    Returns a dict with per-pair ALS values and the overall mean ALS.
    """
    from src.igloo.data_processing import (
        CustomDataset, batch_pad_and_normalize,
        _global_scaler_mean, _global_scaler_scale,
    )
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(1)
    alphas = np.linspace(0.0, 1.0, steps + 1)

    # Build seq → y_raw lookup
    seq_to_y = {s: float(y) for s, y in zip(sequences, y_raw)}

    # Pick pairs
    non_wt = [s for s in sequences if s != wt_seq]
    chosen = rng.choice(len(non_wt), size=min(n_pairs, len(non_wt)), replace=False)
    chosen_seqs = [non_wt[i] for i in chosen]

    print(f"\n[Step 3] Computing ALS ({n_pairs} pairs, method='{interpolation_method}')...")

    predictor_model.eval()

    def _predict_from_emb_array(emb_arr: np.ndarray) -> float:
        """Run predictor on a (1, L, D) numpy embedding array, return denorm prediction."""
        dummy = np.zeros(1, dtype=np.float32)
        ds = CustomDataset(emb_arr, dummy)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=batch_pad_and_normalize)
        with torch.no_grad():
            for xb, lengths, _ in loader:
                xb = xb.to(device)
                if pooling_method == 'esm_attention':
                    mask = torch.zeros(xb.size(0), xb.size(1), device=device)
                    for k, l in enumerate(lengths):
                        mask[k, :l] = 1
                    pred_norm = predictor_model(xb, mask).squeeze(-1).cpu().item()
                else:
                    pred_norm = predictor_model(xb).squeeze(-1).cpu().item()
                return pred_norm * y_log_std + y_log_mean

    # Pre-fetch WT embedding
    wt_emb_raw = get_embeddings([wt_seq], layer=esm_layer, batch_size=1,
                                dataset_name=f'{dataset_name}_als_wt',
                                subdir=embeddings_subdir)
    y_wt_pred = _predict_from_emb_array(wt_emb_raw)
    y_wt_actual = seq_to_y.get(wt_seq, np.nan)

    als_per_pair: list[float] = []
    all_alphas_list: list[float] = []
    all_y_actual_list: list[float] = []
    all_y_ideal_list: list[float] = []

    # WT tensor (reused every iteration — embeddings are cached)
    wt_emb_base = torch.tensor(wt_emb_raw[0], dtype=torch.float32)  # (L_wt, D)

    for seq in tqdm(chosen_seqs, desc='  Pairs'):
        seq_emb_raw = get_embeddings([seq], layer=esm_layer, batch_size=1,
                                     dataset_name=f'{dataset_name}_als_seq',
                                     subdir=embeddings_subdir)
        y_seq_pred = _predict_from_emb_array(seq_emb_raw)

        wt_emb = wt_emb_base.clone()
        seq_emb = torch.tensor(seq_emb_raw[0], dtype=torch.float32)

        # Handle length mismatch
        L_wt, L_s = wt_emb.shape[0], seq_emb.shape[0]
        if L_wt != L_s:
            L_max = max(L_wt, L_s)
            pad_w = torch.zeros(L_max - L_wt, wt_emb.shape[1]) if L_wt < L_max else torch.zeros(0, wt_emb.shape[1])
            pad_s = torch.zeros(L_max - L_s, seq_emb.shape[1]) if L_s < L_max else torch.zeros(0, seq_emb.shape[1])
            wt_emb = torch.cat([wt_emb, pad_w], dim=0)
            seq_emb = torch.cat([seq_emb, pad_s], dim=0)

        y_actual_path: list[float] = []
        y_ideal_path: list[float] = []

        if interpolation_method == 'guided_ig':
            path_tensors = build_guided_ig_path(
                wt_emb.unsqueeze(0), seq_emb.unsqueeze(0),
                predictor_model, pooling_method,
                y_log_mean, y_log_std, device, steps,
            )  # list of (1, L, D), length = steps+1
            for k, x in enumerate(path_tensors):
                alpha_val = float(alphas[k])
                interp_arr = x.cpu().numpy()   # (1, L, D)
                y_a = _predict_from_emb_array(interp_arr)
                y_i = float(y_wt_pred) + alpha_val * (float(y_seq_pred) - float(y_wt_pred))
                y_actual_path.append(y_a)
                y_ideal_path.append(y_i)
                all_alphas_list.append(alpha_val)
                all_y_actual_list.append(y_a)
                all_y_ideal_list.append(y_i)
        else:
            for alpha in alphas:
                interp = interpolate_embeddings(wt_emb, seq_emb, float(alpha),
                                                method=interpolation_method)
                interp_arr = interp.unsqueeze(0).numpy()   # (1, L, D)

                y_a = _predict_from_emb_array(interp_arr)
                y_i = float(y_wt_pred) + float(alpha) * (float(y_seq_pred) - float(y_wt_pred))

                y_actual_path.append(y_a)
                y_ideal_path.append(y_i)
                all_alphas_list.append(float(alpha))
                all_y_actual_list.append(y_a)
                all_y_ideal_list.append(y_i)

        pair_als = float(np.mean([(ya - yi) ** 2
                                  for ya, yi in zip(y_actual_path, y_ideal_path)]))
        als_per_pair.append(pair_als)

    overall_als = float(np.mean(als_per_pair))
    print(f"\n  ALS ({interpolation_method}): {overall_als:.6f}  "
          f"(mean over {len(als_per_pair)} paths)")

    return {
        'als_per_pair': als_per_pair,
        'overall_als': overall_als,
        'alphas': all_alphas_list,
        'y_actual': all_y_actual_list,
        'y_ideal': all_y_ideal_list,
    }


def plot_als(als_result: dict, dataset_name: str, output_dir: str,
             interpolation_method: str = 'linear') -> None:
    """Plot actual vs ideal predictions along the interpolation paths."""
    os.makedirs(output_dir, exist_ok=True)

    alphas = np.array(als_result['alphas'])
    y_act = np.array(als_result['y_actual'])
    y_ide = np.array(als_result['y_ideal'])

    # Bin by alpha and compute mean ± std
    bins = np.linspace(0, 1, 21)
    bin_idx = np.digitize(alphas, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_act = np.array([y_act[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                         for b in range(len(bins) - 1)])
    mean_ide = np.array([y_ide[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                         for b in range(len(bins) - 1)])
    std_act = np.array([y_act[bin_idx == b].std() if (bin_idx == b).any() else np.nan
                        for b in range(len(bins) - 1)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean prediction vs ideal along alpha
    axes[0].plot(bin_centers, mean_ide, 'b--', lw=2, label='Ideal (linear)')
    axes[0].plot(bin_centers, mean_act, 'r-', lw=2, label='Actual (model)')
    axes[0].fill_between(bin_centers, mean_act - std_act, mean_act + std_act,
                         alpha=0.2, color='red')
    axes[0].set_xlabel('α (interpolation step)')
    axes[0].set_ylabel('Predicted Activity')
    axes[0].set_title(f'{dataset_name}: Activity Along Path ({interpolation_method})\n'
                      f'ALS = {als_result["overall_als"]:.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: histogram of per-pair ALS values
    axes[1].hist(als_result['als_per_pair'], bins=20, color='salmon', edgecolor='none', alpha=0.85)
    axes[1].axvline(als_result['overall_als'], color='darkred', lw=1.5, linestyle='--',
                    label=f'mean ALS={als_result["overall_als"]:.4f}')
    axes[1].set_xlabel('Per-Pair ALS')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'{dataset_name}: Distribution of Per-Pair ALS')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_ALS_{interpolation_method}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_methods_comparison_pca(
    real_pooled: np.ndarray,
    all_methods_paths: dict,         # method → list of (steps+1, D) arrays
    all_methods_seqs:  dict,         # method → list of chosen variant seqs
    ref_seqs: list[str],             # canonical sequence order (from first method)
    all_sequences: list[str],
    y_raw: np.ndarray,
    wt_seq: str,
    dataset_name: str,
    output_dir: str,
    max_pairs: int = 10,
) -> None:
    """Grid plot: rows = sequence pairs, columns = interpolation methods.

    Each cell shares the same PCA space so paths are directly comparable.
    """
    from sklearn.decomposition import PCA

    methods = list(all_methods_paths.keys())
    n_methods = len(methods)

    # Limit to first max_pairs so the figure stays readable
    pairs_to_show = ref_seqs[:max_pairs]
    n_pairs = len(pairs_to_show)
    if n_pairs == 0:
        return

    # Fit PCA on real data
    pca = PCA(n_components=2, svd_solver='full')
    real_proj = pca.fit_transform(real_pooled.astype(np.float64))
    exp_var = pca.explained_variance_ratio_

    # WT projection
    if wt_seq in all_sequences:
        wt_proj = real_proj[all_sequences.index(wt_seq)]
    else:
        wt_proj = None

    # Axis limits from real data
    margin = 0.05
    xr = real_proj[:, 0].ptp()
    yr = real_proj[:, 1].ptp()
    xlim = (real_proj[:, 0].min() - margin * xr, real_proj[:, 0].max() + margin * xr)
    ylim = (real_proj[:, 1].min() - margin * yr, real_proj[:, 1].max() + margin * yr)

    seq_to_y = {s: float(v) for s, v in zip(all_sequences, y_raw)}
    all_y = np.array([seq_to_y.get(s, float('nan')) for s in pairs_to_show])
    y_finite = all_y[np.isfinite(all_y)]
    y_min = float(y_finite.min()) if len(y_finite) else 0.0
    y_max = float(y_finite.max()) if len(y_finite) else 1.0
    fitness_cmap = plt.cm.RdBu

    # Method colours for path lines
    method_colors = ['darkorange', 'steelblue', 'mediumseagreen', 'orchid', 'tomato']

    cell = 2.8
    fig, axes = plt.subplots(n_pairs, n_methods,
                              figsize=(n_methods * cell, n_pairs * cell),
                              squeeze=False)

    for col, method in enumerate(methods):
        m_paths = all_methods_paths[method]
        m_seqs  = all_methods_seqs[method]
        seq_to_path = {s: p for s, p in zip(m_seqs, m_paths)}
        color = method_colors[col % len(method_colors)]

        for row, seq in enumerate(pairs_to_show):
            ax = axes[row][col]
            ax.scatter(real_proj[:, 0], real_proj[:, 1],
                       s=2, color='lightgrey', alpha=0.4, zorder=1, rasterized=True)

            if seq in seq_to_path:
                path_proj = pca.transform(seq_to_path[seq])
                pts  = path_proj[:, np.newaxis, :]
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                for seg, a in zip(segs, np.linspace(0.15, 1.0, len(segs))):
                    ax.plot(seg[:, 0], seg[:, 1], color=color,
                            lw=1.4, alpha=a, zorder=3, solid_capstyle='round')
                ax.annotate('', xy=path_proj[-1], xytext=path_proj[-2],
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.4),
                            zorder=4)
                var_end = path_proj[-1]
            else:
                var_end = None

            if wt_proj is not None:
                ax.scatter(*wt_proj, marker='*', s=100, color='limegreen',
                           edgecolors='black', linewidths=0.4, zorder=5)

            if var_end is not None:
                var_y = seq_to_y.get(seq, float('nan'))
                norm_y = np.clip((var_y - y_min) / (y_max - y_min + 1e-12), 0, 1)
                ax.scatter(*var_end, s=50, color=fitness_cmap(norm_y),
                           edgecolors='black', linewidths=0.4, zorder=6)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(method, fontsize=8, fontweight='bold')
            if col == 0:
                var_y = seq_to_y.get(seq, float('nan'))
                label = f'fit={var_y:.2f}' if np.isfinite(var_y) else '?'
                ax.set_ylabel(label, fontsize=7)

    fig.text(0.5, 0.01, f'PC1 ({exp_var[0]*100:.1f}%)', ha='center', fontsize=9)
    fig.text(0.01, 0.5, f'PC2 ({exp_var[1]*100:.1f}%)', va='center',
             rotation='vertical', fontsize=9)

    fig.suptitle(
        f'{dataset_name}: Interpolation Path Comparison\n'
        f'Green ★ = WT  |  dot = variant (red=low, blue=high fitness)  |  '
        f'grey = real embedding cloud',
        fontsize=9, y=1.01,
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{dataset_name}_paths_method_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

# ---------------------------------------------------------------------------
# Per-config analysis runner (shared by single-config and --synthetic modes)
# ---------------------------------------------------------------------------

def _run_one_config(config_path: Path, args, output_root: Path | None = None) -> None:
    """Run the full embedding distribution analysis for one config file.

    Parameters
    ----------
    config_path  : path to a YAML config file.
    args         : parsed argparse namespace (all CLI flags).
    output_root  : if given, results go to output_root/{dataset_name}/;
                   otherwise the config's own results_dir is used.
    """
    import torch  # local to avoid top-level issues when called from sweep

    config = load_config(str(config_path))
    validate_config(config)

    dataset_name   = config['data']['name']
    esm_layer      = config['model']['esm_layer']
    pooling_method = config['model']['pooling_method']
    emb_batch_size = config['model'].get('batch_size', 16)

    if output_root is not None:
        output_dir = str(output_root / dataset_name)
    else:
        results_dir = str(PROJECT_ROOT / config.get(
            'results_dir', f'results/{dataset_name}_dist_analysis'))
        output_dir = os.path.join(results_dir, 'distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)

    embeddings_subdir = 'synthetic' if 'synthetic' in config['data']['path'] else None

    print(f"\n{'='*70}")
    print(f"  Dataset        : {dataset_name}")
    print(f"  ESM layer      : {esm_layer}")
    print(f"  Output         : {output_dir}")
    print(f"{'='*70}")

    # ── Load data ────────────────────────────────────────────────────────────
    data_path = str(PROJECT_ROOT / config['data']['path'])
    df, sequences, y_raw, y, y_log_mean, y_log_std = load_and_prepare_data(config, data_path)
    wt_seq, _ = prepare_wildtype_sequence(df, config)

    # ── Build extreme-sequence lists ─────────────────────────────────────────
    extreme_seqs: list[str] = []
    fit_top_set:  set[str]  = set()
    fit_bot_set:  set[str]  = set()
    if args.include_extremes:
        n_ext = args.n_extremes
        order = np.argsort(y_raw)
        fit_bot_set = {sequences[i] for i in order[:n_ext]  if sequences[i] != wt_seq}
        fit_top_set = {sequences[i] for i in order[-n_ext:] if sequences[i] != wt_seq}
        extreme_seqs = [s for s in
                        [sequences[i] for i in np.concatenate([order[:n_ext], order[-n_ext:]])]
                        if s != wt_seq]
        print(f"  Extreme sequences: {len(extreme_seqs)} "
              f"({n_ext} bottom + {n_ext} top by target value, WT excluded)")

    # Ensure the config's own IG interpolation method is always included in the
    # analysis, even if not explicitly requested via --interp-methods.
    cfg_interp = config.get('integrated_gradients', {}).get('interpolation_path', None)
    if cfg_interp and cfg_interp not in args.interp_methods:
        print(f"  [config] Adding config interpolation_path '{cfg_interp}' to analysis methods.")
        args.interp_methods = list(args.interp_methods) + [cfg_interp]

    # ── Get all dataset embeddings ───────────────────────────────────────────
    # The IG pipeline caches embeddings under "{dataset_name}_integrated_gradients"
    # with config['model']['batch_size'] — use that directly to hit the cache.
    emb_cache_name = f"{dataset_name}_integrated_gradients"
    _emb_bs = emb_batch_size  # from config['model']['batch_size']
    _emb_cache_dir = PROJECT_ROOT / 'data' / 'embeddings'
    if embeddings_subdir:
        _emb_cache_dir = _emb_cache_dir / embeddings_subdir

    # Find the highest contiguous sequence index covered by cache files so we
    # don't trigger ESM recomputation for sequences (e.g. test set) that were
    # never embedded by the IG pipeline.
    _n_seqs = len(sequences)
    _cached_end = 0
    for _i in range(0, _n_seqs, _emb_bs):
        _e = min(_i + _emb_bs, _n_seqs)
        _f = _emb_cache_dir / f"{emb_cache_name}_embeddings_batch_{_i}_{_e-1}.safetensors"
        if _f.exists():
            _cached_end = _e
        else:
            break  # first gap — stop here
    if _cached_end < _n_seqs:
        print(f"  [cache] Cache covers {_cached_end}/{_n_seqs} sequences "
              f"(IG pipeline likely only embedded training set). "
              f"Restricting analysis to cached sequences.")
        sequences  = sequences[:_cached_end]
        y_raw      = y_raw[:_cached_end]
        y          = y[:_cached_end]
        df         = df.iloc[:_cached_end].reset_index(drop=True)
        # Re-filter extreme sets to only include sequences still in scope
        _seq_set   = set(sequences)
        extreme_seqs = [s for s in extreme_seqs if s in _seq_set]
        fit_top_set  = fit_top_set  & _seq_set
        fit_bot_set  = fit_bot_set  & _seq_set

    print(f"\nFetching ESM embeddings for all {len(sequences)} sequences...")
    all_embs  = get_embeddings(sequences, layer=esm_layer, batch_size=_emb_bs,
                               dataset_name=emb_cache_name, subdir=embeddings_subdir)
    pooled_embs = mean_pool(all_embs)

    # ── Build embedding-extremes list ────────────────────────────────────────
    emb_extreme_seqs: list[str] = []
    if args.embedding_extremes:
        centroid = pooled_embs.mean(axis=0)
        dists = np.linalg.norm(pooled_embs - centroid, axis=1)
        order_ee = np.argsort(dists)[::-1]
        emb_extreme_seqs = [sequences[i] for i in order_ee
                            if sequences[i] != wt_seq][:args.n_embedding_extremes]
        print(f"  Embedding-extreme sequences: {len(emb_extreme_seqs)} selected")

    print(f"  Embedding shape (raw): {all_embs.shape}")
    print(f"  Pooled shape:          {pooled_embs.shape}")

    # ── Step 1: Data distribution ────────────────────────────────────────────
    plot_data_distribution(pooled_embs, dataset_name, output_dir,
                           subset_size=args.subset_size)
    plot_all_dimensions_kde(pooled_embs, dataset_name, output_dir)

    # ── Load trained model (for ALS / guided_ig) ─────────────────────────────
    predictor_model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.run_als or 'guided_ig' in args.interp_methods:
        # In synthetic mode, look for the pre-trained noise_0.0 model first
        candidate_pth = (PROJECT_ROOT / 'results' / 'synthetic' /
                         dataset_name / 'noise_0.0' / f'{dataset_name}_training_model.pth')
        try:
            if candidate_pth.exists():
                print(f"\nLoading pre-trained model from {candidate_pth} ...")
                from src.igloo.utils import create_model
                model_obj = create_model(config, device)
                import torch as _torch
                model_obj.load_state_dict(_torch.load(str(candidate_pth), map_location=device))
                model_obj.eval()
                predictor_model = model_obj
                print("  Model loaded successfully.")
            else:
                from src.igloo.train_and_eval import train_or_load_model
                results_dir_for_model = str(PROJECT_ROOT / config.get(
                    'results_dir', f'results/{dataset_name}_dist_analysis'))
                print(f"\nLoading/training model for ALS (device={device})...")
                final_model, _, _ = train_or_load_model(
                    config, sequences, y, results_dir_for_model, device)
                predictor_model = final_model
                print("  Model loaded successfully.")
        except Exception as exc:
            print(f"\n  Warning: Could not load trained model — skipping ALS/guided_ig. ({exc})")

    # ── Steps 2 & 3: FED and ALS per interpolation method ───────────────────
    fed_results:   dict[str, float]        = {}
    fed_per_pair:       dict[str, list[float]]  = {}
    fed_chosen_seqs:    list[str] = []
    all_methods_paths:  dict[str, list[np.ndarray]] = {}   # method → per_pair_paths
    all_methods_seqs:   dict[str, list[str]] = {}          # method → chosen_seqs

    for method in args.interp_methods:
        print(f"\n{'='*60}")
        print(f" Interpolation method: {method}")
        print(f"{'='*60}")

        if method == 'guided_ig' and predictor_model is None:
            print("  Skipping guided_ig — no trained model available.")
            continue

        combined_extra: list[str] = []
        if args.include_extremes:
            combined_extra.extend(extreme_seqs)
        if args.embedding_extremes:
            combined_extra.extend(emb_extreme_seqs)
        seen: set[str] = set()
        combined_extra = [s for s in combined_extra if not (s in seen or seen.add(s))]

        n_random = 0 if args.embedding_extremes else args.n_pairs

        path_embs, per_pair_paths, chosen_seqs = collect_path_embeddings(
            sequences=sequences, wt_seq=wt_seq, n_pairs=n_random,
            steps=args.steps, esm_layer=esm_layer, dataset_name=dataset_name,
            interpolation_method=method, embeddings_subdir=embeddings_subdir,
            extra_seqs=combined_extra if combined_extra else None,
            predictor_model=predictor_model, pooling_method=pooling_method,
            y_log_mean=y_log_mean, y_log_std=y_log_std, device=device,
            train_pooled_embs=pooled_embs,
            precomputed_embs=all_embs,
        )

        fed_val = compute_and_plot_fed(pooled_embs, path_embs,
                                       dataset_name, output_dir, method)
        fed_results[method] = fed_val

        per_pair_feds = compute_per_pair_feds(pooled_embs, per_pair_paths)
        fed_per_pair[method] = per_pair_feds
        if not fed_chosen_seqs:
            fed_chosen_seqs = list(chosen_seqs)

        all_methods_paths[method] = per_pair_paths
        all_methods_seqs[method]  = list(chosen_seqs)

        plot_individual_paths_pca(
            real_pooled=pooled_embs, per_pair_paths=per_pair_paths,
            chosen_seqs=chosen_seqs, all_sequences=sequences,
            y_raw=y_raw, wt_seq=wt_seq, dataset_name=dataset_name,
            output_dir=output_dir, interpolation_method=method,
            emb_extreme_set=set(emb_extreme_seqs) if args.embedding_extremes else None,
            fit_top_set=fit_top_set if args.include_extremes else None,
            fit_bot_set=fit_bot_set if args.include_extremes else None,
            per_pair_feds=per_pair_feds,
        )

        if args.run_als:
            if predictor_model is not None:
                als_result = compute_als(
                    sequences=sequences, y_raw=y_raw, wt_seq=wt_seq,
                    n_pairs=args.n_pairs, steps=args.steps, esm_layer=esm_layer,
                    dataset_name=dataset_name, predictor_model=predictor_model,
                    pooling_method=pooling_method, y_log_mean=y_log_mean,
                    y_log_std=y_log_std, device=device, interpolation_method=method,
                    embeddings_subdir=embeddings_subdir,
                )
                plot_als(als_result, dataset_name, output_dir, method)
            else:
                print("  [Step 3] Skipping ALS (no trained model).")

    # ── Cross-method path comparison PCA ──────────────────────────────────────
    if len(all_methods_paths) > 1:
        print("\n  Plotting cross-method path comparison PCA...")
        # Use the sequence set from the first method (seed=0 → same seqs for all)
        ref_method = next(iter(all_methods_paths))
        ref_seqs   = all_methods_seqs[ref_method]
        plot_methods_comparison_pca(
            real_pooled=pooled_embs,
            all_methods_paths=all_methods_paths,
            all_methods_seqs=all_methods_seqs,
            ref_seqs=ref_seqs,
            all_sequences=sequences,
            y_raw=y_raw,
            wt_seq=wt_seq,
            dataset_name=dataset_name,
            output_dir=output_dir,
        )

    # ── FED boxplot ──────────────────────────────────────────────────────────
    if fed_per_pair:
        print("\n  Plotting per-variant FED boxplot...")
        plot_fed_boxplot(fed_per_pair, dataset_name, output_dir,
                         chosen_seqs=fed_chosen_seqs)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {dataset_name}  (N={len(sequences)}, D={pooled_embs.shape[1]})")
    print(f"{'='*60}")
    for method, fed in fed_results.items():
        pair_feds = fed_per_pair.get(method, [])
        extra = (f"  per-pair μ={np.mean(pair_feds):.4f}  σ={np.std(pair_feds):.4f}"
                 f"  (n={len(pair_feds)})" if pair_feds else "")
        print(f"  FED [{method:20s}]: {fed:.4f}{extra}")
    print(f"\n  Output saved to: {output_dir}")

    import pandas as _pd
    summary_df = _pd.DataFrame([{'method': m, 'FED': v} for m, v in fed_results.items()])
    csv_path = os.path.join(output_dir, f'{dataset_name}_distribution_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"  Summary CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze embedding distribution and interpolation path quality.'
    )
    parser.add_argument(
        'config', nargs='?', default=None,
        help='Path to a YAML/JSON config file (omit when using --synthetic).'
    )
    # ── Synthetic batch mode ─────────────────────────────────────────────────
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Batch mode: loop over all configs in configs/synthetic/ and '
             'analyse every protein × fitness-model combination.'
    )
    parser.add_argument(
        '--proteins', nargs='+', default=None,
        choices=['GB1', 'T4Lysozyme', 'p53'],
        help='Restrict --synthetic mode to specific proteins (default: all).'
    )
    parser.add_argument(
        '--fitness-models', nargs='+', default=None,
        choices=['additive', 'epistatic', 'nonlinear'],
        help='Restrict --synthetic mode to specific fitness models (default: all).'
    )
    parser.add_argument(
        '--synthetic-output-dir', type=str,
        default='results/synthetic/distribution_analysis',
        help='Root output directory for --synthetic mode '
             '(default: results/synthetic/distribution_analysis).'
    )
    # ── Common analysis options ──────────────────────────────────────────────
    parser.add_argument(
        '--n-pairs', type=int, default=20,
        help='Number of sequence pairs to use for path analysis (default: 20).'
    )
    parser.add_argument(
        '--steps', type=int, default=50,
        help='Number of interpolation steps per pair (default: 50).'
    )
    parser.add_argument(
        '--interp-methods', nargs='+',
        default=['linear', 'slerp', 'nearest_neighbor'],
        choices=['linear', 'slerp', 'nearest_neighbor', 'guided_ig'],
        help='Interpolation methods to evaluate (default: linear slerp nearest_neighbor). '
             'guided_ig requires a trained model.'
    )
    parser.add_argument(
        '--run-als', action='store_true',
        help='Run the ALS computation (requires a trained model; skipped by default).'
    )
    parser.add_argument(
        '--subset-size', type=int, default=500,
        help='Number of sequences used for pairwise distance histograms (default: 500).'
    )
    parser.add_argument(
        '--include-extremes', action='store_true',
        help='Always include the top-N and bottom-N fitness sequences as path pairs.'
    )
    parser.add_argument(
        '--n-extremes', type=int, default=10,
        help='Number of top and bottom sequences to include when --include-extremes '
             'is set (default: 10 each end).'
    )
    parser.add_argument(
        '--embedding-extremes', action='store_true',
        help='Select path endpoints from sequences farthest from the dataset centroid.'
    )
    parser.add_argument(
        '--n-embedding-extremes', type=int, default=20,
        help='Number of sequences to select when --embedding-extremes is set (default: 20).'
    )
    args = parser.parse_args()

    # ── Dispatch ─────────────────────────────────────────────────────────────
    if args.synthetic:
        synthetic_config_dir = PROJECT_ROOT / 'configs' / 'synthetic'
        all_configs = sorted(synthetic_config_dir.glob('*.yaml'))
        if not all_configs:
            print(f"No configs found in {synthetic_config_dir}")
            return

        # Optional filtering
        def _matches(p: Path) -> bool:
            name = p.stem  # e.g. GB1_additive_config
            if args.proteins and not any(name.startswith(prot + '_') for prot in args.proteins):
                return False
            if args.fitness_models and not any(f'_{fm}_' in name or name.endswith(f'_{fm}_config')
                                                for fm in args.fitness_models):
                return False
            return True

        configs_to_run = [c for c in all_configs if _matches(c)]
        if not configs_to_run:
            print("No configs matched the specified --proteins / --fitness-models filters.")
            return

        output_root = PROJECT_ROOT / args.synthetic_output_dir
        print(f"Running --synthetic mode: {len(configs_to_run)} config(s) → {output_root}")
        for cfg_path in configs_to_run:
            try:
                _run_one_config(cfg_path, args, output_root=output_root)
            except Exception as exc:
                print(f"\n  ERROR processing {cfg_path.name}: {exc}")
                import traceback; traceback.print_exc()
                print("  Continuing with next config...")

    elif args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        _run_one_config(config_path, args, output_root=None)

    else:
        parser.error("Provide a config file path or use --synthetic to run all proteins.")


if __name__ == '__main__':
    main()

