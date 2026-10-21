#!/usr/bin/env python3
"""
Baseline Attribution Methods for Protein Fitness Prediction
============================================================

Computes three attribution baselines that can be compared against the full
IGLOO Integrated Gradients method.  All three produce output files with the
same schema as ``run_integrated_gradients.py``, so ``evaluate_synthetic_ig.py``
can be run directly on their output directories.

Baselines
---------
1. **gradient_x_input** – Single-point gradient × (variant − WT) embedding
   difference.  No path integration: tests whether the integral step of IGLOO
   adds value over a standard saliency map.

2. **input_difference** – Per-position L2 norm of (variant − WT) embedding
   difference.  Purely sequence/embedding-derived signal with no model
   involvement; tests whether the model is needed at all.

3. **attention_weights** – Per-position attention score extracted from the
   pLM (ESM-2 / ESM-C).  Attention is aggregated across all heads and layers
   and compared between variant and WT.  Tests whether pLM attention alone is
   a sufficient proxy for residue importance.

Output layout
-------------
For a config with ``results_dir: results/synthetic/GB1_additive/noise_0.0/linear``
(or any arbitrary path), the three baselines are written alongside each other:

    {results_dir}/gradient_x_input/
        {name}_integrated_gradients_results.csv
        {name}_position_importance_summary.csv
    {results_dir}/input_difference/
        {name}_integrated_gradients_results.csv
        {name}_position_importance_summary.csv
    {results_dir}/attention_weights/
        {name}_integrated_gradients_results.csv
        {name}_position_importance_summary.csv

This mirrors the IGLOO directory structure so ``evaluate_synthetic_ig.py``
discovers them when given the top-level ``results/synthetic/`` root.

Usage
-----
    source IGLOO_env/bin/activate

    # Run all three baselines
    python scripts/pipeline/run_baseline_attributions.py configs/GB1_additive_config.yaml

    # Run a subset of baselines
    python scripts/pipeline/run_baseline_attributions.py configs/GB1_additive_config.yaml \\
        --baselines gradient_x_input input_difference

    # Apply ad-hoc config overrides (same syntax as run_integrated_gradients.py)
    python scripts/pipeline/run_baseline_attributions.py configs/GB1_additive_config.yaml \\
        training.noise.label_noise_std=0.1 results_dir=results/myjob

    # Validate config only
    python scripts/pipeline/run_baseline_attributions.py configs/GB1_additive_config.yaml \\
        --validate-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.config_utils import (
    load_config,
    select_sequences_for_analysis,
    validate_config,
)
from src.igloo.data_processing import (
    _global_scaler_mean,
    _global_scaler_scale,
    load_and_prepare_data,
    prepare_wildtype_sequence,
)
from src.igloo.embedding import get_embeddings
from src.igloo.esm_adapter import load_esm_backend
from src.igloo.plot import (
    plot_averaged_gradient_effects_comparison,
    plot_gradients_heatmap,
    plot_integrated_gradients,
    plot_signed_gradient_effects,
    plot_variants_vs_wt_gradients,
)
from src.igloo.train_and_eval import train_or_load_model
from src.igloo.utils import evaluate_model_performance

# ---------------------------------------------------------------------------
# Helpers shared with run_integrated_gradients.py
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Merge ``key.subkey=value`` strings into *cfg* in-place."""
    def _cast(v: str):
        if v.lower() in ('true', 'false'):
            return v.lower() == 'true'
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    for item in overrides:
        if '=' not in item:
            print(f"Warning: ignoring malformed override (no '='): {item}")
            continue
        dotpath, _, raw_val = item.partition('=')
        keys = dotpath.split('.')
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _cast(raw_val)


class _Tee:
    """Duplicate writes to both the original stdout and a file."""
    def __init__(self, original, file_obj):
        self._orig = original
        self._file = file_obj
    def write(self, data):
        self._orig.write(data)
        self._file.write(data)
    def flush(self):
        self._orig.flush()
        self._file.flush()
    def __getattr__(self, name):
        return getattr(self._orig, name)


# ---------------------------------------------------------------------------
# Normalisation helper (mirrors the pipeline used during training)
# ---------------------------------------------------------------------------

def _normalize_embedding(emb_np: np.ndarray) -> np.ndarray:
    """Apply the same global standardisation used during training.

    *emb_np* is shape ``(L, D)``.  Returns ``(L, D)`` float32.
    """
    from src.igloo.data_processing import _global_scaler_mean, _global_scaler_scale
    flat = emb_np.reshape(-1)
    if _global_scaler_mean is not None and _global_scaler_scale is not None:
        flat_normed = (flat - _global_scaler_mean) / (_global_scaler_scale + 1e-8)
    else:
        flat_normed = (flat - flat.mean()) / (flat.std() + 1e-8)
    return flat_normed.reshape(emb_np.shape).astype(np.float32)


def _predict(model, norm_emb: np.ndarray, device: torch.device,
             pooling_method: str) -> float:
    """Run a single normalised embedding through the trained model."""
    model.eval()
    x = torch.tensor(norm_emb, dtype=torch.float32, device=device).unsqueeze(0)  # (1, L, D)
    with torch.no_grad():
        if pooling_method == 'esm_attention':
            mask = torch.ones(1, x.shape[1], device=device)
            pred = model(x, mask)
        else:
            pred = model(x)
    return float(pred.squeeze().cpu())


# ---------------------------------------------------------------------------
# Baseline 1 – Gradient × Input
# ---------------------------------------------------------------------------

def _gradient_x_input(
    variant_norm: np.ndarray,
    baseline_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    pooling_method: str,
    attribution_method: str = 'signed',
) -> np.ndarray:
    """Single-point gradient × (variant − baseline) saliency map.

    Computes the gradient of the model output with respect to the *variant*
    embedding at the variant point itself, then multiplies element-wise by
    the difference from the baseline.  The per-position attribution is then
    summed over the embedding dimension (signed) or L2-normed (magnitude),
    matching the convention used by the IG signed/magnitude modes.

    Returns
    -------
    (L,) float32 array of per-position attributions.
    """
    model.eval()
    x_var = torch.tensor(variant_norm, dtype=torch.float32, device=device).unsqueeze(0)
    x_var.requires_grad_(True)

    if pooling_method == 'esm_attention':
        mask = torch.ones(1, x_var.shape[1], device=device)
        pred = model(x_var, mask)
    else:
        pred = model(x_var)

    pred.sum().backward()

    grad = x_var.grad.squeeze(0).detach().cpu().numpy()      # (L, D)
    diff = (variant_norm - baseline_norm)                     # (L, D)
    attr = grad * diff                                        # (L, D)
    if attribution_method == 'signed':
        return attr.sum(axis=-1).astype(np.float32)           # (L,) — preserves sign
    else:
        return np.linalg.norm(attr, axis=-1).astype(np.float32)  # (L,) — always ≥ 0


# ---------------------------------------------------------------------------
# Baseline 2 – Input Difference
# ---------------------------------------------------------------------------

def _input_difference(
    variant_norm: np.ndarray,
    baseline_norm: np.ndarray,
    attribution_method: str = 'signed',
) -> np.ndarray:
    """Per-position difference of (variant − baseline) embedding.

    For signed mode: sums over the embedding dimension to preserve direction.
    For magnitude mode: L2 norm (always ≥ 0).
    No model involved — purely sequence/embedding-derived.

    Returns
    -------
    (L,) float32 array of per-position differences.
    """
    diff = variant_norm - baseline_norm      # (L, D)
    if attribution_method == 'signed':
        return diff.sum(axis=-1).astype(np.float32)
    else:
        return np.linalg.norm(diff, axis=-1).astype(np.float32)  # (L,)


# ---------------------------------------------------------------------------
# Baseline 3 – Attention Weights
# ---------------------------------------------------------------------------

class _AttentionCapturer:
    """Register forward hooks on every attention module to capture weights.

    Works for both ESM-2 (``MultiheadAttention`` sub-modules with
    ``in_proj_weight``) and ESM-C (``MultiHeadAttention`` blocks in
    ``transformer.blocks``).

    After a forward pass, ``self.attentions`` is a list of tensors:
        ESM-2 : (n_heads, L+2, L+2) – includes BOS/EOS special tokens
        ESM-C : (n_heads, L+2, L+2) – typically same

    Call ``remove()`` to deregister all hooks.
    """

    def __init__(self, esm_model: torch.nn.Module, device: torch.device):
        self.attentions: list[torch.Tensor] = []
        self._hooks: list = []
        self._device = device
        self._register(esm_model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attn_hook(self, module, input_, output):
        """Generic hook: tries to extract the attention weight tensor."""
        # torch.nn.MultiheadAttention returns (output, attn_weights) when
        # need_weights=True and average_attn_weights=False.
        if isinstance(output, tuple):
            attn = output[1]
            if attn is not None:
                self.attentions.append(attn.detach().cpu())
                return
        # Some implementations (ESM-C) store weights in module attributes
        for attr in ('attn_weights', '_last_attn_weights', 'attention_weights'):
            if hasattr(module, attr):
                w = getattr(module, attr)
                if w is not None:
                    self.attentions.append(w.detach().cpu())
                    return

    def _register(self, esm_model: torch.nn.Module) -> None:
        """Walk the model and attach hooks to recognised attention modules."""
        mods_hooked = 0
        for name, module in esm_model.named_modules():
            cls_name = type(module).__name__
            if 'MultiheadAttention' in cls_name or 'MultiHeadAttention' in cls_name:
                # ESM-2: make sure need_weights is enabled
                if hasattr(module, 'need_weights'):
                    module.need_weights = True    # type: ignore[assignment]
                    module.average_attn_weights = False  # type: ignore[assignment]
                h = module.register_forward_hook(self._attn_hook)
                self._hooks.append(h)
                mods_hooked += 1
        if mods_hooked == 0:
            print("  [AttentionCapturer] Warning: no MultiheadAttention modules "
                  "found — attention baseline will produce zero attributions.")

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _attention_weights_esm2(
    sequence: str,
    esm_backend,
    device: torch.device,
    layer: int,
) -> np.ndarray | None:
    """Extract per-position attention importance from an ESM-2 model.

    Returns mean over all heads of the *column sum* of the attention matrix
    at *layer* (how much total attention each position receives from all
    others).  BOS/EOS tokens are stripped.  Returns ``(L,)`` float32 or None.
    """
    from esm.sdk.api import ESMProtein  # noqa: F401 – confirm ESM-2 path
    # ESM-2 path
    model  = esm_backend.model
    alph   = esm_backend.alphabet
    batch_converter = alph.get_batch_converter()
    _, _, tokens = batch_converter([("prot", sequence)])
    tokens = tokens.to(device)

    # Capture attention for all layers
    capturer = _AttentionCapturer(model, device)
    with torch.no_grad():
        model(tokens, repr_layers=[layer], return_contacts=False,
              need_head_weights=True)
    capturer.remove()

    if not capturer.attentions:
        return None

    # Stack: (n_layers, batch, n_heads, L+2, L+2) or (batch, n_heads, L+2, L+2)
    # Use only the requested layer if possible, else average all
    attn_list = capturer.attentions
    L2 = attn_list[0].shape[-1]  # L + 2 (BOS + EOS)
    L  = L2 - 2

    # Stack all captured tensors → mean over layers and heads
    stacked = []
    for a in attn_list:
        # a shape: (batch, n_heads, S, S) or (n_heads, S, S)
        while a.dim() > 3:
            a = a.squeeze(0)
        stacked.append(a)  # (n_heads, S, S)

    mat = torch.stack(stacked, dim=0)   # (n_layers, n_heads, S, S)
    # Mean over layers and heads → (S, S)
    mat_mean = mat.mean(dim=(0, 1))     # (S, S)
    # Column sum = how much attention position j receives; strip BOS(0)/EOS(-1)
    col_sum = mat_mean[:, 1:-1].sum(dim=0).numpy()  # (L,)
    # Normalise to [0, 1]
    rng = col_sum.max() - col_sum.min()
    if rng > 0:
        col_sum = (col_sum - col_sum.min()) / rng
    return col_sum.astype(np.float32)


def _attention_weights_esmc(
    sequence: str,
    esm_backend,
    device: torch.device,
) -> np.ndarray | None:
    """Extract per-position attention importance from an ESM-C model via hooks.

    ESM-C does not expose attention weights through its public encode() API,
    so we register forward hooks on the internal MultiHeadAttention modules.
    """
    from esm.sdk.api import ESMProtein  # type: ignore[import]

    model = esm_backend.model

    capturer = _AttentionCapturer(model, device)
    protein = ESMProtein(sequence=sequence)

    # Build raw input tensors via the tokeniser
    from esm.tokenization import EsmSequenceTokenizer  # type: ignore[import]
    tokenizer = EsmSequenceTokenizer()
    tokens = tokenizer([sequence])
    if isinstance(tokens, dict):
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            model(**tokens)
    else:
        tokens = tokens.to(device)
        with torch.no_grad():
            model(tokens)

    capturer.remove()

    if not capturer.attentions:
        # Fall back: return zero vector so caller can still produce output
        return np.zeros(len(sequence), dtype=np.float32)

    L = len(sequence)
    stacked = []
    for a in capturer.attentions:
        while a.dim() > 3:
            a = a.squeeze(0)
        # Trim or pad to match L (+2 for BOS/EOS)
        S = a.shape[-1]
        stacked.append(a[:, :S, :S])

    mat = torch.stack(stacked, dim=0).mean(dim=(0, 1))   # (S, S)
    # Strip BOS/EOS if present (S = L+2)
    if mat.shape[-1] == L + 2:
        col_sum = mat[:, 1:-1].sum(dim=0).numpy()
    else:
        col_sum = mat.sum(dim=0).numpy()[:L]

    rng = col_sum.max() - col_sum.min()
    if rng > 0:
        col_sum = (col_sum - col_sum.min()) / rng
    return col_sum.astype(np.float32)


def _get_attention_attribution(
    variant_seq: str,
    baseline_seq: str,
    esm_backend,
    device: torch.device,
    layer: int,
) -> np.ndarray:
    """Attention-based attribution = |attn(variant) − attn(baseline)| per position.

    Returns ``(L,)`` float32.  Falls back to zero vector on any failure.
    """
    L = len(variant_seq)
    try:
        if esm_backend.family == 'esm2':
            attn_var  = _attention_weights_esm2(variant_seq,  esm_backend, device, layer)
            attn_base = _attention_weights_esm2(baseline_seq, esm_backend, device, layer)
        else:
            attn_var  = _attention_weights_esmc(variant_seq,  esm_backend, device)
            attn_base = _attention_weights_esmc(baseline_seq, esm_backend, device)

        if attn_var is None or attn_base is None:
            return np.zeros(L, dtype=np.float32)
        L_ = min(len(attn_var), len(attn_base), L)
        diff = np.abs(attn_var[:L_] - attn_base[:L_])
        result = np.zeros(L, dtype=np.float32)
        result[:L_] = diff
        return result
    except Exception as exc:
        print(f"    [attention] failed for sequence (len={L}): {exc}")
        return np.zeros(L, dtype=np.float32)


# ---------------------------------------------------------------------------
# Result serialisation (mirrors utils.py create_visualizations_and_exports)
# ---------------------------------------------------------------------------

def _make_results_row(
    seq_name: str,
    positions: list[int],    # 1-indexed
    amino_acids: list[str],
    attr: np.ndarray,        # per-position attribution, same length as positions
    predicted_activity: float,
    baseline_activity: float,
    effect_vs_baseline: float,
    actual_activity,
    baseline_actual_activity,
) -> list[dict]:
    rows = []
    for pos, aa, val in zip(positions, amino_acids, attr):
        rows.append({
            'sequence_name':              seq_name,
            'position':                   int(pos),
            'amino_acid':                 aa,
            'integrated_gradient':        float(val),
            'abs_gradient':               float(abs(val)),
            'predicted_activity':         float(predicted_activity),
            'baseline_activity':          float(baseline_activity),
            'effect_vs_baseline':         float(effect_vs_baseline),
            'effect_vs_baseline_unclipped': float(effect_vs_baseline),
            'actual_activity':            actual_activity,
            'baseline_actual_activity':   baseline_actual_activity,
        })
    return rows


def _save_results(
    all_rows: list[dict],
    out_dir: str,
    dataset_name: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, f'{dataset_name}_integrated_gradients_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved results CSV → {csv_path}")

    # Position importance summary
    if results_df.empty:
        print("  [Warning] No result rows – skipping position importance CSV.")
        return

    pos_imp = results_df.groupby('position').agg(
        mean_gradient=('integrated_gradient', 'mean'),
        std_gradient=('integrated_gradient', 'std'),
        mean_abs_gradient=('abs_gradient', 'mean'),
        max_abs_gradient=('abs_gradient', 'max'),
    ).round(6)
    pos_imp['importance_score'] = pos_imp['mean_abs_gradient'] + pos_imp['std_gradient'].fillna(0)
    pos_imp = pos_imp.sort_values('importance_score', ascending=False)

    imp_path = os.path.join(out_dir, f'{dataset_name}_position_importance_summary.csv')
    pos_imp.to_csv(imp_path)
    print(f"  Saved importance CSV → {imp_path}")


# ---------------------------------------------------------------------------
# Main runner class
# ---------------------------------------------------------------------------

SUPPORTED_BASELINES = ('gradient_x_input', 'input_difference', 'attention_weights')


class BaselineAttributionsRunner:
    """Mirrors the interface of IntegratedGradientsRunner for baseline methods."""

    def __init__(
        self,
        config_path: str,
        overrides: list[str] | None = None,
        baselines: tuple[str, ...] = SUPPORTED_BASELINES,
    ):
        import random
        self.config = load_config(config_path)
        if overrides:
            _apply_overrides(self.config, overrides)
        validate_config(self.config)
        self.baselines = [b for b in baselines if b in SUPPORTED_BASELINES]
        if not self.baselines:
            raise ValueError(f"No valid baselines specified.  Choose from: {SUPPORTED_BASELINES}")

        seed = self.config.get('training', {}).get('seed', 42)
        try:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception as e:
            print(f"[Warning] Could not set all random seeds: {e}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Populated during run
        self.df             = None
        self.sequences      = None
        self.y              = None
        self.y_raw          = None
        self.y_log_mean     = None
        self.y_log_std      = None
        self.wt_seq         = None
        self.final_model    = None
        self.trainval_idx   = None
        self.test_idx       = None
        self.esm_backend    = None

    # ------------------------------------------------------------------ #
    # Initialisation steps (reuse igloo helpers)
    # ------------------------------------------------------------------ #

    def load_and_prepare_data(self):
        data_path = str(_PROJECT_ROOT / self.config['data']['path'])
        result = load_and_prepare_data(self.config, data_path)
        self.df, self.sequences, self.y_raw, self.y, self.y_log_mean, self.y_log_std = result

    def inject_label_noise(self):
        noise_cfg = self.config.get('training', {}).get('noise', {})
        noise_std = float(noise_cfg.get('label_noise_std', 0.0))
        if noise_std <= 0.0:
            return
        seed = int(noise_cfg.get('label_noise_seed', 0))
        rng  = np.random.default_rng(seed)
        eps  = rng.standard_normal(len(self.y_raw)).astype(np.float32) * noise_std
        self.y_raw = self.y_raw + eps
        self.y     = self.y + (eps / self.y_log_std)
        print(f"Label noise injected — std={noise_std:.4f} (seed={seed})")

    def split_data(self):
        training_cfg  = self.config.get('training', {})
        test_fraction = float(training_cfg.get('test_fraction', 0.0))
        if test_fraction <= 0.0:
            self.trainval_idx = np.arange(len(self.sequences))
            self.test_idx     = np.array([], dtype=int)
            return
        split_seed = int(training_cfg.get('split_seed', 42))
        from igloo.data_processing import split_dataset
        self.trainval_idx, self.test_idx = split_dataset(
            self.df, test_fraction=test_fraction, seed=split_seed
        )
        print(f"Data split: {len(self.trainval_idx)} train+val, "
              f"{len(self.test_idx)} test")

    def prepare_wildtype(self):
        self.wt_seq, _ = prepare_wildtype_sequence(self.df, self.config)

    def _trainval_sequences(self):
        if self.trainval_idx is None:
            return self.sequences
        return [self.sequences[i] for i in self.trainval_idx]

    def _trainval_y(self):
        if self.trainval_idx is None:
            return self.y
        return self.y[self.trainval_idx]

    def train_or_load(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        import yaml
        snap = os.path.join(
            results_dir,
            f"{self.config['data']['name']}_config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        with open(snap, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        self.final_model, _, _ = train_or_load_model(
            self.config, self._trainval_sequences(), self._trainval_y(),
            results_dir, self.device
        )

    def load_esm_backend(self):
        model_cfg      = self.config.get('model', {})
        esm_model_name = model_cfg.get('esm_model_name', 'esmc_300m')
        print(f"Loading ESM backend ({esm_model_name})…")
        self.esm_backend = load_esm_backend(esm_model_name, self.device)

    def _ensure_esm_backend(self):
        """Lazily load the ESM backend if not already loaded."""
        if self.esm_backend is None:
            print("  [ESM] Backend not yet loaded — loading now (cache miss or attention baseline)…")
            self.load_esm_backend()

    # ------------------------------------------------------------------ #
    # Embedding cache helpers
    # ------------------------------------------------------------------ #

    def _build_embedding_lookup(self) -> dict[str, np.ndarray]:
        """Pre-load all embeddings into a seq→(L,D) numpy dict."""
        model_cfg       = self.config.get('model', {})
        esm_layer       = model_cfg['esm_layer']
        embeddings_subdir = 'synthetic' if 'synthetic' in self.config.get('data', {}).get('path', '') else None
        dataset_name    = self.config['data']['name']
        batch_size      = model_cfg.get('batch_size', 16)

        esm_model_name  = model_cfg.get('esm_model_name', 'esmc_300m')
        tv_seqs = self._trainval_sequences()
        # Include model name in cache key so different pLMs use separate caches
        _cache_prefix = f"{dataset_name}_{esm_model_name}"
        cache_name = f"{_cache_prefix}_integrated_gradients"
        # get_embeddings will use the cache; only calls esm_backend on misses.
        # We pass None and let it raise if the cache is incomplete — caller
        # should ensure backend is loaded before calling this if misses are expected.
        try:
            emb_np = get_embeddings(
                tv_seqs, layer=esm_layer, batch_size=batch_size,
                dataset_name=cache_name, subdir=embeddings_subdir,
                esm_backend=self.esm_backend,
            )  # (N, L, D)
        except Exception:
            self._ensure_esm_backend()
            emb_np = get_embeddings(
                tv_seqs, layer=esm_layer, batch_size=batch_size,
                dataset_name=cache_name, subdir=embeddings_subdir,
                esm_backend=self.esm_backend,
            )  # (N, L, D)
        lookup: dict[str, np.ndarray] = {seq: emb_np[i] for i, seq in enumerate(tv_seqs)}

        # Test set
        if self.test_idx is not None and len(self.test_idx) > 0:
            test_seqs  = [self.sequences[i] for i in self.test_idx]
            test_cache = f"{_cache_prefix}_integrated_gradients_test"
            test_emb   = get_embeddings(
                test_seqs, layer=esm_layer, batch_size=batch_size,
                dataset_name=test_cache, subdir=embeddings_subdir,
                esm_backend=self.esm_backend,
            )
            for i, seq in enumerate(test_seqs):
                lookup[seq] = test_emb[i]

        print(f"Embedding lookup ready: {len(lookup)} sequences")
        return lookup

    # ------------------------------------------------------------------ #
    # Sequence selection (mirrors IntegratedGradientsRunner)
    # ------------------------------------------------------------------ #

    def _select_sequences(self) -> list[tuple[str, str, str]]:
        _default_sel = {
            'include_extremes': True,
            'include_median':   True,
            'include_top_n':    0,
            'include_bottom_n': 0,
            'custom_sequences': [],
            'max_error_percent': 10.0,
        }
        sel_cfg = (
            self.config.get('analysis', {}).get('sequence_selection')
            or self.config.get('sequence_selection')
            or _default_sel
        )
        seqs = select_sequences_for_analysis(
            self.df, self.y_raw, self.wt_seq, sel_cfg,
            model=self.final_model, sequences=self.sequences, y_pred=None
        )

        has_split = (self.test_idx is not None and len(self.test_idx) > 0)
        if has_split:
            test_set = {self.sequences[i] for i in self.test_idx}
            annotated = []
            for name, seq, baseline in seqs:
                label = 'test' if seq in test_set else 'trainval'
                annotated.append((f"{name}[{label}]", seq, baseline))
            seqs = annotated

        # Always add poly-alanine
        poly_ala = 'A' * len(self.wt_seq)
        seqs.append(('PolyAlanine', poly_ala, self.wt_seq))
        return seqs

    # ------------------------------------------------------------------ #
    # Per-baseline computation
    # ------------------------------------------------------------------ #

    def _compute_one(
        self,
        method: str,
        sequences_to_analyze: list[tuple[str, str, str]],
        embedding_lookup: dict[str, np.ndarray],
        out_dir: str,
    ) -> None:
        """Compute one baseline method and save results."""
        dataset_name    = self.config['data']['name']
        model_cfg       = self.config.get('model', {})
        pooling_method  = model_cfg.get('pooling_method', 'mean')
        esm_layer       = model_cfg['esm_layer']
        esm_backend     = self.esm_backend
        attribution_method = self.config.get('integrated_gradients', {}).get(
            'attribution_method', 'signed'
        )

        print(f"\n── Baseline: {method} ─────────────────────────────────")

        all_rows: list[dict] = []
        n_failed = 0

        for seq_name, variant_seq, baseline_seq in sequences_to_analyze:
            L = len(variant_seq)
            try:
                # Retrieve or compute embeddings
                if variant_seq in embedding_lookup:
                    var_emb = embedding_lookup[variant_seq][:L]   # (L, D)
                else:
                    self._ensure_esm_backend()
                    var_emb = self.esm_backend.get_sequence_embeddings(
                        [variant_seq], layer=esm_layer)[0][:L]

                if baseline_seq in embedding_lookup:
                    base_emb = embedding_lookup[baseline_seq][:L]
                else:
                    self._ensure_esm_backend()
                    base_emb = self.esm_backend.get_sequence_embeddings(
                        [baseline_seq], layer=esm_layer)[0][:L]

                var_norm  = _normalize_embedding(var_emb)
                base_norm = _normalize_embedding(base_emb)

                pred_act  = _predict(self.final_model, var_norm,  self.device, pooling_method)
                base_act  = _predict(self.final_model, base_norm, self.device, pooling_method)
                effect    = pred_act - base_act

                # Look up actual activity
                try:
                    idx         = self.sequences.index(variant_seq)
                    actual_act  = float(self.y_raw[idx])
                except ValueError:
                    actual_act  = 'N/A'
                try:
                    bidx        = self.sequences.index(baseline_seq)
                    base_actual = float(self.y_raw[bidx])
                except ValueError:
                    base_actual = 'N/A'

                # ── Compute attribution ──────────────────────────────
                if method == 'gradient_x_input':
                    attr = _gradient_x_input(
                        var_norm, base_norm, self.final_model, self.device,
                        pooling_method, attribution_method
                    )
                elif method == 'input_difference':
                    attr = _input_difference(var_norm, base_norm, attribution_method)
                elif method == 'attention_weights':
                    self._ensure_esm_backend()
                    attr = _get_attention_attribution(
                        variant_seq, baseline_seq, self.esm_backend, self.device, esm_layer
                    )
                else:
                    raise ValueError(f"Unknown baseline: {method}")

                # Pad / trim attr to length L
                if len(attr) < L:
                    attr = np.concatenate([attr, np.zeros(L - len(attr), dtype=np.float32)])
                attr = attr[:L]

                positions   = list(range(1, L + 1))
                amino_acids = list(variant_seq)

                all_rows.extend(_make_results_row(
                    seq_name, positions, amino_acids, attr,
                    pred_act, base_act, effect, actual_act, base_actual
                ))
                print(f"  {seq_name}: pred={pred_act:.4f} base={base_act:.4f} "
                      f"Δ={effect:+.4f}  max_attr={attr.max():.4f}")

            except Exception as exc:
                print(f"  [Error] {seq_name}: {exc}")
                traceback.print_exc()
                n_failed += 1

        if n_failed:
            print(f"  {n_failed} sequences failed and were skipped.")

        _save_results(all_rows, out_dir, dataset_name)
        self._generate_figures(all_rows, out_dir, dataset_name, method)

    # ------------------------------------------------------------------ #
    # Figure generation
    # ------------------------------------------------------------------ #

    def _zip_per_sequence_dirs(self, results_dir: str) -> None:
        """Zip any *_per_sequence directories in results_dir and remove the originals."""
        import glob
        import shutil
        import zipfile

        pattern = os.path.join(results_dir, '*_per_sequence')
        dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]
        if not dirs:
            return
        for dir_path in dirs:
            zip_path = dir_path + '.zip'
            print(f"  Zipping {os.path.basename(dir_path)} → {os.path.basename(zip_path)} …",
                  end=' ', flush=True)
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(dir_path):
                    for fname in files:
                        fpath   = os.path.join(root, fname)
                        arcname = os.path.relpath(fpath, os.path.dirname(dir_path))
                        zf.write(fpath, arcname)
            shutil.rmtree(dir_path)
            print("done")

    def _generate_figures(
        self,
        all_rows: list[dict],
        out_dir: str,
        dataset_name: str,
        method: str,
    ) -> None:
        """Generate the same set of figures as run_integrated_gradients.py."""
        if not all_rows:
            print("  [figures] No rows — skipping figure generation.")
            return

        attribution_method = self.config.get('integrated_gradients', {}).get(
            'attribution_method', 'signed'
        )

        # Reconstruct ig_results dict expected by plotting functions
        ig_results: dict = {}
        for row in all_rows:
            name = row['sequence_name']
            if name not in ig_results:
                ig_results[name] = {
                    'positions':           [],
                    'amino_acids':         [],
                    'position_gradients':  [],
                    'predicted_activity':  row['predicted_activity'],
                    'baseline_activity':   row['baseline_activity'],
                    'effect_vs_baseline':  row['effect_vs_baseline'],
                    'effect_vs_baseline_unclipped': row.get('effect_vs_baseline_unclipped',
                                                            row['effect_vs_baseline']),
                    'actual_activity':     row.get('actual_activity', 'N/A'),
                    'baseline_actual_activity': row.get('baseline_actual_activity', 'N/A'),
                }
            ig_results[name]['positions'].append(row['position'])
            ig_results[name]['amino_acids'].append(row['amino_acid'])
            ig_results[name]['position_gradients'].append(row['integrated_gradient'])

        # Convert lists to numpy arrays so plotting functions can do arithmetic on them
        for entry in ig_results.values():
            entry['position_gradients'] = np.array(entry['position_gradients'], dtype=np.float32)
            entry['positions']          = np.array(entry['positions'],          dtype=np.int32)
            entry['amino_acids']        = np.array(entry['amino_acids'])

        results_df = pd.DataFrame(all_rows)

        print(f"  Generating figures in {out_dir} …")

        # 1. Per-sequence bar charts
        try:
            ig_plot_path = os.path.join(out_dir, f'{dataset_name}_integrated_gradients_analysis.png')
            plot_integrated_gradients(ig_results, ig_plot_path)
            print(f"  Saved IG analysis plot → {ig_plot_path}")
        except Exception as e:
            print(f"  [figures] plot_integrated_gradients failed: {e}")

        # 2. Heatmap
        try:
            heatmap_path = os.path.join(out_dir, f'{dataset_name}_integrated_gradients_heatmap.png')
            plot_gradients_heatmap(
                results_df,
                f'{dataset_name} {method} Attribution Heatmap Across Sequences',
                heatmap_path,
            )
            print(f"  Saved heatmap → {heatmap_path}")
        except Exception as e:
            print(f"  [figures] plot_gradients_heatmap failed: {e}")

        # 3. Variant vs WT gradient difference bars
        try:
            comparison_path = os.path.join(out_dir, f'{dataset_name}_variants_vs_WT_gradients.png')
            plot_variants_vs_wt_gradients(ig_results, comparison_path)
            print(f"  Saved variant comparison plot → {comparison_path}")
        except Exception as e:
            print(f"  [figures] plot_variants_vs_wt_gradients failed: {e}")

        # 4 & 5. Signed-gradient plots (same attribution_method check as IG script)
        if attribution_method == 'signed':
            try:
                signed_path = os.path.join(out_dir, f'{dataset_name}_signed_gradient_effects.png')
                plot_signed_gradient_effects(ig_results, signed_path, top_n=15)
                print(f"  Saved signed effects plot → {signed_path}")
            except Exception as e:
                print(f"  [figures] plot_signed_gradient_effects failed: {e}")

            if len(ig_results) >= 4:
                try:
                    top_vs_bottom_path = os.path.join(
                        out_dir, f'{dataset_name}_top_vs_bottom_effects.png'
                    )
                    n = min(3, len(ig_results) // 2)
                    plot_averaged_gradient_effects_comparison(
                        ig_results, top_vs_bottom_path, top_n=n, bottom_n=n
                    )
                    print(f"  Saved top-vs-bottom plot → {top_vs_bottom_path}")
                except Exception as e:
                    print(f"  [figures] plot_averaged_gradient_effects_comparison failed: {e}")

        # Zip per-sequence plot directories (mirrors IG script behaviour)
        self._zip_per_sequence_dirs(out_dir)

    # ------------------------------------------------------------------ #
    # Top-level run
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Execute the full pipeline: data → model → baselines → CSVs."""
        # Determine the directory where the trained model lives.
        # model_dir in config == noise-level dir, e.g.
        #   results/synthetic/GB1_additive/noise_0.0
        # Baseline outputs are written as siblings of the interpolation-method
        # dirs (linear/, slerp/, …) so that evaluate_synthetic_ig.py discovers
        # them automatically.
        results_dir_cfg = self.config.get(
            'results_dir', f"results/{self.config['data']['name']}_ig"
        )
        model_dir_cfg = self.config.get('model_dir', results_dir_cfg)
        noise_level_dir = str(_PROJECT_ROOT / model_dir_cfg)

        # We still need a results_dir to satisfy train_or_load (CV plots etc.)
        # Reuse the interpolation-method dir specified in the config so we
        # don't create new artefacts.
        ig_results_dir = str(_PROJECT_ROOT / results_dir_cfg)

        # Baseline outputs land in {noise_level_dir}/{method}/
        base_results_dir = noise_level_dir
        os.makedirs(base_results_dir, exist_ok=True)

        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(
            base_results_dir,
            f"{self.config['data']['name']}_baselines_run_log_{ts}.txt"
        )
        _log_file    = open(log_path, 'w')
        _orig_stdout = sys.stdout
        sys.stdout   = _Tee(_orig_stdout, _log_file)

        try:
            t0 = time.perf_counter()
            print("=" * 60)
            print("BASELINE ATTRIBUTIONS PIPELINE")
            print("=" * 60)
            print(f"Device          : {self.device}")
            print(f"Dataset         : {self.config['data']['name']}")
            print(f"Baselines       : {', '.join(self.baselines)}")
            print(f"Results root    : {base_results_dir}")
            print("=" * 60)

            # 1. Data
            print("\n[1/6] Loading data…")
            self.load_and_prepare_data()
            self.inject_label_noise()
            self.split_data()

            # 2. Wildtype
            print("\n[2/6] Preparing wildtype sequence…")
            self.prepare_wildtype()

            # 3. Train or load model
            # Pass the IG results dir so that any CV plots land there;
            # train_or_load_model resolves the .pth path via model_dir in config.
            print("\n[3/6] Loading pre-trained model (force_retrain=False)…")
            self.config['force_retrain'] = False  # never re-train when run as baseline
            self.train_or_load(ig_results_dir)

            # 4. ESM backend — only needed if attention_weights is requested
            # or if embeddings are missing from cache.
            needs_esm = ('attention_weights' in self.baselines)
            if needs_esm:
                print("\n[4/6] Loading ESM backend (required for attention_weights)…")
                self.load_esm_backend()
            else:
                print("\n[4/6] Skipping ESM backend load (not needed for selected baselines).")

            # 5. Embedding lookup
            print("\n[5/6] Building embedding lookup…")
            # If cache hits cover all sequences, ESM backend is never called.
            if self.esm_backend is None and not needs_esm:
                # We still need the backend as a fallback for cache misses.
                # Try to detect whether all sequences are cached before loading.
                pass  # _build_embedding_lookup will load backend lazily if needed
            embedding_lookup = self._build_embedding_lookup()

            # 6. Sequence selection
            sequences_to_analyze = self._select_sequences()
            print(f"\nSelected {len(sequences_to_analyze)} sequences for analysis.")

            # 7. Run each baseline
            print("\n[6/6] Computing baselines…")
            for method in self.baselines:
                out_dir = os.path.join(base_results_dir, method)
                self._compute_one(method, sequences_to_analyze, embedding_lookup, out_dir)

            elapsed = time.perf_counter() - t0
            print(f"\n{'='*60}")
            print(f"All baselines complete in {elapsed:.1f}s")
            print(f"Results written to sub-directories of: {base_results_dir}")
            print(f"{'='*60}")

        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = _orig_stdout
            _log_file.close()
        print(f"Log saved to {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute attribution baselines (gradient×input, input difference, "
                    "attention weights) using the same config as run_integrated_gradients.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # All three baselines
    python scripts/pipeline/run_baseline_attributions.py configs/GB1_additive_config.yaml

    # Subset of baselines
    python scripts/pipeline/run_baseline_attributions.py configs/GB1_additive_config.yaml \\
        --baselines gradient_x_input input_difference

    # With config overrides
    python scripts/pipeline/run_baseline_attributions.py configs/GB1_additive_config.yaml \\
        training.noise.label_noise_std=0.1 results_dir=results/myjob
        """,
    )
    parser.add_argument('config_path', help='Path to YAML/JSON config file')
    parser.add_argument(
        '--baselines',
        nargs='+',
        default=list(SUPPORTED_BASELINES),
        choices=list(SUPPORTED_BASELINES),
        metavar='BASELINE',
        help='Which baselines to run (default: all three). '
             f'Choices: {", ".join(SUPPORTED_BASELINES)}',
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the config file, do not run analysis.',
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        metavar='key.subkey=value',
        help='Dot-path config overrides, e.g. results_dir=results/myjob',
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: config file not found: {args.config_path}")
        sys.exit(1)

    cfg = load_config(args.config_path)
    if args.overrides:
        _apply_overrides(cfg, args.overrides)
    try:
        validate_config(cfg)
    except Exception as e:
        print(f"Config validation failed: {e}")
        sys.exit(1)

    print(f"Config '{args.config_path}' is valid.")
    if args.validate_only:
        return

    runner = BaselineAttributionsRunner(
        args.config_path,
        overrides=args.overrides,
        baselines=tuple(args.baselines),
    )
    runner.run()


if __name__ == '__main__':
    main()
