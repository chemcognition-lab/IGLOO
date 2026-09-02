"""
IGLOO playground utilities (standalone / Colab edition).

Thin, transparent wrappers around the vendored `src/igloo` package for
interactive, step-by-step inspection of the embedding -> attention-pooling
-> integrated-gradients pipeline for GB1 additive / Linear interpolation.

Design goal: every function here EXPOSES intermediate tensors (shapes,
values) instead of hiding them inside one black-box call the way the
production scripts (scripts/run_integrated_gradients.py) do. Nothing here
is a new algorithm -- it's the same math as src/igloo/sequence_ig.py's
linear-IG branch, just unpacked for inspection.

This file is identical in behaviour to the in-repo IGLOO/colab version,
except for one line: PROJECT_ROOT is computed relative to *this folder*
(the standalone package root) rather than a parent IGLOO/ checkout, since
this copy vendors its own src/igloo, data/, and results/ subtrees so it
runs with zero dependency on the original repo layout (e.g. on Colab).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# This file lives at the standalone package root, alongside src/, data/,
# results/, scripts/ -- so PROJECT_ROOT is just this file's own directory
# (contrast with the in-repo colab/ version, where it's two levels up).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.igloo.embedding import get_embeddings  # noqa: E402
from src.igloo.utils import create_model  # noqa: E402
from src.igloo.sequence_ig import interpolate_embeddings  # noqa: E402
from src.igloo.ig_metrics import (  # noqa: E402
    _build_importance_vec,
    _wt_sequence,
    _load_gt_json,
    compute_metrics_for_dir,
    discover_result_dirs_multiseed,
    _ndcg_1d,
    _PROTEIN_DEFS,
    _FIT_FNS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM_MODEL_NAME = "esmc_300m"
ESM_LAYER = 33
EMBEDDINGS_SUBDIR = "synthetic"

# GB1 additive uses normalization: "none" in its config -> identity target rescale.
Y_LOG_MEAN = 0.0
Y_LOG_STD = 1.0


# ---------------------------------------------------------------------------
# Step 1: load a trained model checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(model_dir: "str | Path", dataset_name: str = "GB1_additive",
                     device: torch.device = DEVICE):
    """Load a trained model checkpoint (no training, no embeddings needed).

    Returns (model, model_config, normalization_params).
    """
    ckpt_path = Path(model_dir) / f"{dataset_name}_training_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. This standalone bundle only ships "
            f"the noise=0.0/seed_1 run -- pick that cell in Section 1, or run "
            f"scripts/run_integrated_gradients.py yourself (see README) to "
            f"generate a different (noise, seed) cell."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_model(ckpt["model_config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  model_config          = {ckpt['model_config']}")
    print(f"  normalization_params  = {ckpt['normalization_params']}")
    print(f"  trained for epoch     = {ckpt.get('epoch')}")
    print(f"  final train/val loss  = {ckpt.get('train_loss')}, {ckpt.get('val_loss')}")
    print(f"  total parameters      = {n_params:,}")
    return model, ckpt["model_config"], ckpt["normalization_params"]


# ---------------------------------------------------------------------------
# Step 2: embeddings (cached -- instant unless the sequence was never seen before)
# ---------------------------------------------------------------------------
def get_sequence_embedding(sequence: str, dataset_name: str = "GB1_additive_playground",
                            layer: int = ESM_LAYER, subdir: str = EMBEDDINGS_SUBDIR,
                            device: torch.device = DEVICE) -> np.ndarray:
    """Embed one sequence through the frozen ESM-C-300M backbone (layer 33).

    Returns a (1, L, D) numpy array. Hits data/embeddings/ cache on repeat calls
    with the *same sequence*; a genuinely new sequence triggers one fresh ESM
    forward pass (GPU recommended, CPU works but slower).

    IMPORTANT: get_embeddings() names its cache file from `dataset_name` plus
    the sequence's *position within this call's list* (always 0 here, since
    we call it one sequence at a time) -- NOT from the sequence's content. So
    two different sequences embedded with the same dataset_name would collide
    and silently return the first sequence's cached embedding for both. To
    make this function safe to call repeatedly with different sequences, we
    fold a hash of the sequence into the cache key automatically.
    """
    seq_tag = hashlib.sha1(sequence.encode()).hexdigest()[:10]
    unique_dataset_name = f"{dataset_name}_{seq_tag}"
    emb = get_embeddings(
        [sequence], layer=layer, batch_size=1, dataset_name=unique_dataset_name,
        subdir=subdir, esm_model_name=ESM_MODEL_NAME, device=device,
    )
    return emb  # (1, L, D)


def get_dataset_embeddings(sequences: list[str], dataset_name: str, batch_size: int = 64,
                            layer: int = ESM_LAYER, subdir: str = EMBEDDINGS_SUBDIR,
                            device: torch.device = DEVICE) -> np.ndarray:
    """Bulk-embed a list of sequences. Pass the EXACT dataset_name + batch_size +
    sequence order used by the original training run to hit the existing cache
    (see `describe_embedding_cache_key` below) -- otherwise this recomputes.
    """
    return get_embeddings(
        sequences, layer=layer, batch_size=batch_size, dataset_name=dataset_name,
        subdir=subdir, esm_model_name=ESM_MODEL_NAME, device=device,
    )


def describe_embedding_cache_key(protein_dataset_name: str = "GB1_additive",
                                  model_batch_size: int = 64) -> str:
    """The exact dataset_name the production pipeline used for its trainval
    embedding cache (see src/igloo/integrated_gradients.py), so bulk calls
    above can hit the same cache instead of recomputing."""
    model_tag = ESM_MODEL_NAME.lower().replace("-", "_")
    return f"{protein_dataset_name}_{model_tag}_integrated_gradients"


# ---------------------------------------------------------------------------
# Step 3: normalization + forward pass (exposing attention weights)
# ---------------------------------------------------------------------------
def normalize_embedding(emb: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    """Replicates sequence_ig.py's _normalize: flatten -> z-score -> reshape.
    emb: (B, L, D) -> same shape, differentiable if emb.requires_grad."""
    b, l, d = emb.shape
    flat = emb.reshape(b, -1)
    flat_norm = (flat - mean) / scale
    return flat_norm.reshape(b, l, d)


def model_forward(model: torch.nn.Module, norm_emb: torch.Tensor,
                   pooling_method: str = "esm_attention",
                   y_log_mean: float = Y_LOG_MEAN, y_log_std: float = Y_LOG_STD) -> torch.Tensor:
    """Forward pass on already-normalised embeddings -> rescaled prediction (B, 1)."""
    b, l, _ = norm_emb.shape
    if pooling_method == "esm_attention":
        mask = torch.ones(b, l, device=norm_emb.device)
        out = model(norm_emb, mask)
    else:
        out = model(norm_emb)
    return out * y_log_std + y_log_mean


def get_attention_pooling_weights(model: torch.nn.Module, norm_emb: torch.Tensor) -> torch.Tensor:
    """Replicates Attention1d.forward up to (but not past) the softmax, so you
    can see WHICH residue positions the attention-pooling layer weighs most
    heavily for a given (normalised) input -- independent of the downstream MLP.
    norm_emb: (B, L, D) -> returns (B, L) attention weights summing to 1 per row.
    """
    with torch.no_grad():
        b, l, _ = norm_emb.shape
        attn = model.attention1d.layer(norm_emb)   # (B, L, 1) masked conv score per position
        attn = attn.view(b, -1)                    # (B, L)
        attn = torch.softmax(attn, dim=-1)
        return attn


# ---------------------------------------------------------------------------
# Step 4: integrated gradients, linear path, step by step (matches sequence_ig.py)
# ---------------------------------------------------------------------------
def compute_linear_ig_stepwise(
    model: torch.nn.Module, input_emb: np.ndarray, baseline_emb: np.ndarray,
    mean: float, scale: float, pooling_method: str = "esm_attention",
    y_log_mean: float = Y_LOG_MEAN, y_log_std: float = Y_LOG_STD,
    steps: int = 500, batch_size: int = 32, device: torch.device = DEVICE,
    verbose: bool = True,
) -> dict:
    """Faithful, transparent re-implementation of the production linear-IG
    computation (src/igloo/sequence_ig.py lines ~1194-1254, trapezoidal rule),
    unpacked so every intermediate tensor is inspectable.

    input_emb, baseline_emb: (1, L, D) numpy arrays, e.g. from get_sequence_embedding.
    Returns a dict with signed_ig (L,), magnitude_ig (L,), path_predictions,
    a completeness-axiom check, and the shapes seen along the way.
    """
    input_t = torch.tensor(input_emb, dtype=torch.float32, device=device)
    baseline_t = torch.tensor(baseline_emb, dtype=torch.float32, device=device)
    L, D = input_t.shape[1], input_t.shape[2]

    alphas = torch.linspace(0, 1, steps + 1, device=device)
    trap_w = torch.ones(steps + 1, device=device)
    trap_w[0] = 0.5
    trap_w[-1] = 0.5  # trapezoidal rule: endpoints weighted half

    embed_diff = (input_t - baseline_t).squeeze(0)  # (L, D), constant tangent for a linear path

    acc_signed = torch.zeros(L, device=device)
    acc_mag = torch.zeros(L, D, device=device)

    interp_batch_shape = None
    grad_batch_shape = None
    all_predictions = []

    n_batches = (len(alphas) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(alphas), batch_size)):
        batch_alphas = alphas[i:i + batch_size]
        batch_w = trap_w[i:i + batch_size]

        batch_embs = [
            interpolate_embeddings(baseline_t, input_t, a.item(), method="linear")
            for a in batch_alphas
        ]
        batch_tensor = torch.cat(batch_embs, dim=0)  # (B, L, D)
        if interp_batch_shape is None:
            interp_batch_shape = tuple(batch_tensor.shape)

        batch_req = batch_tensor.detach().requires_grad_(True)
        norm_batch = normalize_embedding(batch_req, mean, scale)
        out = model_forward(model, norm_batch, pooling_method, y_log_mean, y_log_std)
        all_predictions.extend(out.detach().cpu().numpy().flatten().tolist())

        grad_batch = torch.autograd.grad(out.sum(), batch_req, create_graph=False)[0]  # (B, L, D)
        if grad_batch_shape is None:
            grad_batch_shape = tuple(grad_batch.shape)

        w = batch_w.view(-1, 1, 1)                          # (B, 1, 1)
        acc_mag += (grad_batch * w).sum(dim=0)               # (L, D)
        tangents = embed_diff.unsqueeze(0).expand_as(grad_batch)
        acc_signed += (grad_batch * tangents * w).sum(dim=-1).sum(dim=0)  # (L,)

        if verbose and bi == 0:
            print(f"Interpolation path: {steps + 1} alpha steps in [0, 1], "
                  f"processed in {n_batches} batches of <= {batch_size}")
            print(f"  interpolated-embedding batch shape : {interp_batch_shape}  (B, L, D)")
            print(f"  gradient batch shape                : {grad_batch_shape}  (B, L, D), same as input")

    acc_signed /= steps
    acc_mag /= steps
    all_predictions = np.array(all_predictions)

    if verbose:
        print(f"  model prediction along path: F(baseline)={all_predictions[0]:.4f} "
              f"-> F(input)={all_predictions[-1]:.4f}")
        print(f"  final signed IG vector shape        : {tuple(acc_signed.shape)}  (one scalar per residue position)")
        print(f"  final |IG| (pre-norm) accum shape   : {tuple(acc_mag.shape)}  (L, D) -> normed to (L,) below")

    signed_ig = acc_signed.detach().cpu().numpy()
    magnitude_ig = acc_mag.norm(dim=-1).detach().cpu().numpy()
    completeness_gap = float(signed_ig.sum() - (all_predictions[-1] - all_predictions[0]))

    return {
        "signed_ig": signed_ig,                 # (L,)
        "magnitude_ig": magnitude_ig,            # (L,)
        "path_predictions": all_predictions,     # (steps+1,)
        "sum_signed_ig": float(signed_ig.sum()),
        "f_input_minus_f_baseline": float(all_predictions[-1] - all_predictions[0]),
        "completeness_gap": completeness_gap,    # should be ~0 (completeness axiom)
    }


# ---------------------------------------------------------------------------
# Ground truth + official (precomputed) attribution lookups
# ---------------------------------------------------------------------------
def load_ground_truth(protein: str = "GB1"):
    """Returns (importance_vector, annotation_mask, by_category_dict, background_positions)."""
    return _build_importance_vec(protein)


def load_wt_sequence(protein: str = "GB1") -> str:
    return _wt_sequence(protein)


def load_official_ig_for_sequence(results_csv: "str | Path", sequence_name: str):
    """Pull the already-computed (production-pipeline) IG values for one
    sequence out of a *_integrated_gradients_results.csv, for cross-checking
    against compute_linear_ig_stepwise()'s reimplementation above."""
    import pandas as pd
    df = pd.read_csv(results_csv)
    return df[df["sequence_name"] == sequence_name].sort_values("position")


# ---------------------------------------------------------------------------
# Sparsity / dimensionality diagnostics -- for the "is the embedding space
# sparse, would a different pooling/reduction technique help" hypothesis
# ---------------------------------------------------------------------------
def embedding_sparsity_report(emb: np.ndarray, near_zero_thresh: float = 1e-3, seed: int = 0) -> dict:
    """
    emb: (N, L, D) or (M, D) array of RAW (pre-normalisation) embeddings.
    Returns per-dimension near-zero fraction and PCA explained-variance curve,
    to check whether the D=960 embedding dimensions are meaningfully
    independent or whether most sequence-to-sequence variance concentrates
    in a small subspace (which would argue for a different pooling/reduction
    technique than plain attention-weighted averaging).
    """
    flat = emb.reshape(-1, emb.shape[-1]) if emb.ndim == 3 else emb
    frac_near_zero = (np.abs(flat) < near_zero_thresh).mean(axis=0)  # (D,)

    centered = flat - flat.mean(axis=0, keepdims=True)
    if centered.shape[0] > 4000:
        idx = np.random.default_rng(seed).choice(centered.shape[0], 4000, replace=False)
        centered = centered[idx]
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    explained_var_ratio = (s ** 2) / (s ** 2).sum()
    cum_explained = np.cumsum(explained_var_ratio)
    n_dims_for_90pct = int(np.searchsorted(cum_explained, 0.90) + 1)
    n_dims_for_50pct = int(np.searchsorted(cum_explained, 0.50) + 1)

    return {
        "n_vectors": flat.shape[0],
        "d_embedding": flat.shape[1],
        "mean_frac_near_zero_per_dim": float(frac_near_zero.mean()),
        "max_frac_near_zero_per_dim": float(frac_near_zero.max()),
        "frac_near_zero_per_dim": frac_near_zero,
        "explained_var_ratio": explained_var_ratio,
        "cum_explained_var": cum_explained,
        "n_dims_for_50pct_variance": n_dims_for_50pct,
        "n_dims_for_90pct_variance": n_dims_for_90pct,
    }


# ---------------------------------------------------------------------------
# Mutation-level ground truth: leave-one-out (LOO) fitness contributions
# ---------------------------------------------------------------------------
def compute_loo_contributions(
    protein: str, fitness_model: str, wt_seq: str, mut_list: list[tuple[int, str, str]],
) -> list[float]:
    """For one sequence's list of mutations [(position, wt_aa, mut_aa), ...],
    compute each mutation's leave-one-out (LOO) fitness contribution using the
    TRUE synthetic fitness function (additive/epistatic/nonlinear) -- i.e. how
    much fitness changes when *only that one mutation* is reverted to WT,
    holding the other mutations fixed. This is the exact ground-truth relevance
    signal the mutation-level NDCG metric ranks |IG| against (see
    src/igloo/ig_metrics.py, "Per-sequence mutation NDCG").

    Positive contribution = that mutation was damaging (removing it increased
    fitness); this is what gets clipped to >=0 and used as NDCG relevance.
    """
    prot_def = _PROTEIN_DEFS.get(protein)
    fit_fn = _FIT_FNS.get(fitness_model)
    if prot_def is None or fit_fn is None:
        raise RuntimeError(
            f"No ground-truth fitness function available for protein={protein!r}, "
            f"fitness_model={fitness_model!r} -- src/igloo/synthetic_data.py may not "
            f"have imported correctly."
        )
    seq_full = list(wt_seq)
    for pos, _, mut_aa in mut_list:
        seq_full[pos] = mut_aa
    f_full = fit_fn("".join(seq_full), prot_def, mut_list)

    contributions = []
    for k in range(len(mut_list)):
        rest = [m for i, m in enumerate(mut_list) if i != k]
        seq_rest = list(wt_seq)
        for pos2, _, mut_aa2 in rest:
            seq_rest[pos2] = mut_aa2
        f_rest = fit_fn("".join(seq_rest), prot_def, rest)
        contributions.append(float(f_rest - f_full))
    return contributions
