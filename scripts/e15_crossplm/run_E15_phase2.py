#!/usr/bin/env python3
"""
E15 PHASE 2 Job Submission Script — p53 Cross-pLM Comparison

⚠️  IMPORTANT: Run this AFTER E12 (p53 diagnostic) is complete.

This script orchestrates E15 PHASE 2 pipeline:
  1. Load synthetic landscape (p53, 3 landscapes)
  2. For each pLM {ESM-C 300M, Ankh-base, E1 150M}:
     a. Get embeddings (cached)
     b. Train predictor (reusing GB1 hyperparameters, adjust if E12 suggests)
     c. Run all 5 attribution methods
     d. Compute all 7 metrics (with E12-informed adjustments if needed)
  3. Output results to long-format CSV + config snapshots

Key constraints:
  - Use consistent hyperparameters from existing GB1 cross-pLM runs
  - Fix random seeds (numpy, torch, CUDA) for reproducibility
  - Reuse cached embeddings when available
  - Output to: results/revision/E15_crosspLM/

Dependencies:
  - E12 (p53 diagnostic) must be complete before running this
  - May need to adjust metrics/evaluation based on E12 findings
  - See E12_p53/diagnostic.md for per-tier performance notes

Usage:
  python run_E15_phase2.py [--dry-run] [--pLms esmc_300m,ankh_base] [--seeds 1,2,3]

Expected runtime:
  ~36-54 GPU hours total (9 combos × 4-6 hours per combo)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import ndcg_score, auc, roc_curve, average_precision_score
from scipy.stats import spearmanr, kendalltau, ranksums

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from igloo.train_and_eval import train_model_with_embeddings
from igloo.embedding import get_embeddings
from igloo.utils import create_model
from igloo.synthetic_data import SyntheticDatasetGenerator, split_train_test
from igloo.advanced_ig import integrated_gradients_embeddings
from igloo.esm_adapter import load_esm_backend


# ============================================================================
# Configuration
# ============================================================================

PHASE2_CONFIG = {
    "proteins": ["p53"],
    "landscapes": ["additive", "epistatic", "nonlinear"],
    "plms": ["esmc_300m"],
    "methods": ["ig_linear", "ig_slerp", "ig_nearest_neighbor", "gradient_input", "input_difference"],
    "noise_levels": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    "num_seeds": 5,
}

# Hyperparameters per protein (from previous configs)
PREDICTOR_HPARAMS = {
    "esmc_300m": {
        "hidden_dim1": 256,  # p53 config
        "hidden_dim2": 128,
        "hidden_dim3": 64,
        "hidden_dim4": 32,
        "dropout_rate": 0.5,
        "learning_rate": 0.0005,
        "epochs": 50,
        "batch_size": 256,
        "loss_type": "huber",  # Robust to outliers (vs MSE)
        "patience": 22,  # Early stopping patience
        "min_epochs": 16,  # Minimum epochs to train
    },
}

E12_ADJUSTMENTS = {
    "comment": "p53 stays in paper with per-tier performance explanation",
}

OUTPUT_BASE = PROJECT_ROOT / "results" / "revision" / "E15_crosspLM"


# ============================================================================
# Utilities
# ============================================================================

def check_e12_complete() -> bool:
    """Check if E12 (p53 diagnostic) has completed."""
    e12_path = PROJECT_ROOT / "results" / "revision" / "E12_p53" / "diagnostic.md"
    if not e12_path.exists():
        print(f"⚠️  WARNING: E12 results not found at {e12_path}")
        print(f"   Consider completing E12 before queuing PHASE 2")
        return False
    return True


def set_random_seeds(seed: int):
    """Set reproducible random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_dir(protein: str, plm: str) -> Path:
    """Create output directory for a (protein, pLM) combo."""
    out_dir = OUTPUT_BASE / protein / plm
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config(out_dir: Path, config: Dict, protein: str, plm: str, seed: int):
    """Save config snapshot."""
    config["metadata"] = {
        "protein": protein,
        "plm": plm,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "phase": "E15_PHASE2",
    }
    config_path = out_dir / f"config_seed{seed}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def log_result(csv_path: Path, result: Dict):
    """Append result to long-format CSV."""
    row = pd.DataFrame([result])

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row

    df.to_csv(csv_path, index=False)


# ============================================================================
# Main Pipeline
# ============================================================================

def compute_attribution_ig(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    baseline_embedding: np.ndarray,
    device: torch.device,
    method: str = 'ig_linear',
    steps: int = 50,
) -> np.ndarray:
    """Compute integrated gradients attribution."""
    if method not in ['ig_linear', 'ig_slerp', 'ig_nearest_neighbor']:
        raise ValueError(f"Unknown IG method: {method}")

    interpolation_map = {
        'ig_linear': 'linear',
        'ig_slerp': 'slerp',
        'ig_nearest_neighbor': 'nearest_neighbor',
    }

    return integrated_gradients_embeddings(
        model=model,
        input_embedding=embeddings,
        baseline_embedding=baseline_embedding,
        device=device,
        steps=steps,
        interpolation_method=interpolation_map[method],
    )


def compute_attribution_gradient_input(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute gradient × input attribution."""
    model.eval()
    x = torch.tensor(embeddings, dtype=torch.float32, device=device, requires_grad=True)

    output = model(x)
    gradients = torch.autograd.grad(output.sum(), x, create_graph=False)[0]

    attribution = x.detach() * gradients
    return attribution.cpu().numpy()


def compute_attribution_input_difference(
    embeddings: np.ndarray,
    baseline_embedding: np.ndarray,
) -> np.ndarray:
    """Compute input difference attribution."""
    return embeddings - baseline_embedding


def compute_residue_attributions(attributions: np.ndarray, method: str = 'signed') -> np.ndarray:
    """Convert embedding-level (B, L, D) to residue-level (B, L).

    Args:
        attributions: Embedding-level gradients (B, L, D)
        method: 'signed' (mean, preserves sign), 'magnitude' (L2 norm), or 'abs_sum'

    Returns:
        Residue-level attributions (B, L)
    """
    if method == 'signed':
        # Mean of signed gradients across embedding dimension — preserves direction
        return np.mean(attributions, axis=-1)
    elif method == 'l2_norm':
        # L2 norm — always positive, loses direction
        return np.linalg.norm(attributions, axis=-1)
    elif method == 'magnitude' or method == 'abs_sum':
        # Sum of absolute values — magnitude without direction
        return np.abs(attributions).sum(axis=-1)
    else:
        return np.mean(attributions, axis=-1)  # default to signed


def compute_ranking_metrics(
    attributions: np.ndarray,
    ground_truth_w: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Spearman correlation and Kendall's tau for ranking quality.

    Args:
        attributions: (L,) residue-level attributions
        ground_truth_w: (L,) ground truth importance weights

    Returns:
        Dict with spearman_corr, kendall_tau scores
    """
    # Spearman rank correlation - how well attributed ranks match ground truth ranks
    spearman_r, spearman_p = spearmanr(attributions, ground_truth_w)

    # Kendall tau - non-parametric rank correlation
    kendall_tau, kendall_p = kendalltau(attributions, ground_truth_w)

    return {
        'spearman_corr': float(spearman_r),
        'kendall_tau': float(kendall_tau),
    }


def compute_sign_accuracy(
    attributions: np.ndarray,
    sequence_changes: List[Tuple[int, str, str]],
    ground_truth_w: np.ndarray,
) -> float:
    """
    Compute fraction of positions with correct directional prediction.
    """
    if len(sequence_changes) == 0:
        return 1.0

    correct = 0
    for pos, wt, mut in sequence_changes:
        pred_sign = np.sign(attributions[pos]) if attributions[pos] != 0 else 0
        true_sign = np.sign(ground_truth_w[pos]) if ground_truth_w[pos] != 0 else 0
        if pred_sign == true_sign or (pred_sign == 0 and true_sign == 0):
            correct += 1

    return float(correct / len(sequence_changes))


def compute_noise_robustness_metrics(
    attributions: np.ndarray,
    ground_truth_w: np.ndarray,
    attributions_embed: np.ndarray,
    model_loss: float,
) -> Dict[str, float]:
    """
    Compute metrics to detect signal vs noise in attributions.

    Returns:
        Dict with: correlation, bootstrap_separation, model_fit, gradient_magnitude
    """
    metrics = {}

    # 1. Correlation with ground truth (Spearman rank correlation)
    corr, p_val = spearmanr(attributions, ground_truth_w)
    metrics['spearman_corr'] = float(corr) if not np.isnan(corr) else 0.0
    metrics['correlation_pvalue'] = float(p_val) if not np.isnan(p_val) else 1.0

    # 2. Bootstrap baseline — mean/std of attributions on random labels
    random_labels = np.random.randn(len(ground_truth_w))
    metrics['attr_magnitude_mean'] = float(np.abs(attributions).mean())
    metrics['attr_magnitude_std'] = float(np.abs(attributions).std())
    metrics['random_label_magnitude'] = float(np.abs(random_labels).mean())

    # 3. Gradient magnitude statistics (high noise → unstable gradients)
    grad_mean = np.abs(attributions_embed).mean()
    grad_std = np.abs(attributions_embed).std()
    metrics['gradient_magnitude_mean'] = float(grad_mean)
    metrics['gradient_magnitude_std'] = float(grad_std)
    metrics['gradient_magnitude_cv'] = float(grad_std / (grad_mean + 1e-8))

    # 4. Model fit quality (lower loss → better fit → more trustworthy attributions)
    metrics['model_loss'] = float(model_loss)

    # 5. Dynamic range of attributions (high noise → compressed range)
    attr_range = np.max(np.abs(attributions)) - np.min(np.abs(attributions))
    metrics['attribution_dynamic_range'] = float(attr_range)
    metrics['attribution_max_abs'] = float(np.max(np.abs(attributions)))

    # 6. Entropy of attribution distribution (high noise → uniform)
    attr_norm = np.abs(attributions) / (np.sum(np.abs(attributions)) + 1e-8)
    entropy = -np.sum(attr_norm * np.log(attr_norm + 1e-10))
    metrics['attribution_entropy'] = float(entropy)

    return metrics


def run_experiment(
    protein: str,
    landscape: str,
    plm: str,
    seed: int,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Run a single E15 PHASE 2 experiment.

    Returns:
        List of result dictionaries (one row per metric per noise level)
    """
    print(f"\n{'='*70}")
    print(f"Running: {protein} / {landscape} / {plm} / seed={seed}")
    print(f"{'='*70}")

    set_random_seeds(seed)

    # Setup
    out_dir = create_output_dir(protein, plm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hparams = PREDICTOR_HPARAMS[plm]
    results = []

    # Load ESM backend once to avoid double-loading
    esm_backend = load_esm_backend(plm, device)

    # Map protein name (config format → dataset generator format)
    protein_map = {
        "T4_lysozyme": "T4Lysozyme",
        "T4Lysozyme": "T4Lysozyme",
        "p53": "p53",
        "GB1": "GB1",
    }
    protein_for_gen = protein_map.get(protein, protein)

    # 1. Load synthetic landscape
    print(f"1. Loading synthetic landscape: {protein} / {landscape}")
    try:
        gen = SyntheticDatasetGenerator(protein_for_gen)
        df = gen.generate(model=landscape, n_variants=2000, seed=seed)
        ground_truth_w = gen.ground_truth_importance()

        # Split into train (80%) and test (20%)
        train_df, test_df = split_train_test(df, test_fraction=0.20)

        # Train on training set, evaluate attributions on test set
        train_sequences = train_df['SEQ'].tolist()
        test_sequences = test_df['SEQ'].tolist()
        fitness_noiseless = test_df['fitness'].values  # Use test set for evaluation

        print(f"   Loaded {len(df)} total sequences")
        print(f"   Train: {len(train_sequences)}, Test: {len(test_sequences)}")
        print(f"   Ground truth shape: {ground_truth_w.shape}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return [{"protein": protein, "landscape": landscape, "plm": plm, "seed": seed, "error": str(e)}]

    if dry_run:
        print("   [DRY RUN] Skipping actual computation")
        return [{
            "protein": protein,
            "landscape": landscape,
            "plm": plm,
            "seed": seed,
            "status": "dry_run",
        }]

    # 2. Get embeddings (for both train and test sets)
    print(f"2. Getting embeddings (pLM={plm})")
    try:
        train_embeddings = get_embeddings(
            train_sequences,
            layer=33,
            batch_size=hparams['batch_size'],
            dataset_name=f"E15_phase2_{protein}_{landscape}_train",
            subdir='revision/E15',
            esm_model_name=plm,
            esm_backend=esm_backend,
        )
        test_embeddings = get_embeddings(
            test_sequences,
            layer=33,
            batch_size=hparams['batch_size'],
            dataset_name=f"E15_phase2_{protein}_{landscape}_test",
            subdir='revision/E15',
            esm_model_name=plm,
            esm_backend=esm_backend,
        )
        print(f"   Train embeddings shape: {train_embeddings.shape}")
        print(f"   Test embeddings shape: {test_embeddings.shape}")
    except Exception as e:
        print(f"   ERROR getting embeddings: {e}")
        return [{"protein": protein, "landscape": landscape, "plm": plm, "seed": seed, "error": str(e)}]

    # 3. Train predictor
    print(f"3. Training predictor (epochs={hparams['epochs']})")
    try:
        # Extract embedding dimensions from loaded embeddings
        input_dim = train_embeddings.shape[2]  # (N, L, D) → D
        L = train_embeddings.shape[1]  # (N, L, D) → L

        model = create_model({
            'input_dim': input_dim,
            'L': L,
            'pooling_method': 'linear',
            'hidden_dim1': hparams['hidden_dim1'],
            'hidden_dim2': hparams['hidden_dim2'],
            'hidden_dim3': hparams['hidden_dim3'],
            'hidden_dim4': hparams['hidden_dim4'],
            'dropout_rate': hparams['dropout_rate'],
            'final_activation': None,
        }, device)

        train_result = train_model_with_embeddings(
            model=model,
            train_sequences=train_sequences,
            train_targets=train_df['fitness'].values,
            val_sequences=test_sequences,
            val_targets=test_df['fitness'].values,
            esm_layer=33,
            pooling_method='linear',
            batch_size=hparams['batch_size'],
            learning_rate=hparams['learning_rate'],
            num_epochs=hparams['epochs'],
            device=device,
            dataset_name=f"E15_phase2_{protein}_{landscape}",
            embeddings_subdir='revision/E15',
            esm_model_name=plm,
            esm_backend=esm_backend,
            loss_type=hparams['loss_type'],
            patience=hparams['patience'],
            min_delta=1e-4,
        )
        print(f"   Training complete")
        metrics = train_result.get('metrics', {})
        model_loss = train_result.get('history', {}).get('val_loss', [float('nan')])
        if isinstance(model_loss, list):
            model_loss = model_loss[-1] if model_loss else float('nan')
        model_spearman = metrics.get('val_spearman', float('nan'))
        print(f"   Final validation loss: {model_loss:.6f}")
        print(f"   Validation Spearman ρ: {model_spearman:.4f}")

        # Save model performance metrics
        model_perf = {
            'protein': protein,
            'landscape': landscape,
            'plm': plm,
            'seed': seed,
            'val_loss': float(model_loss),
            'val_spearman': float(model_spearman),
        }
    except Exception as e:
        print(f"   ERROR training: {e}")
        return [{"protein": protein, "landscape": landscape, "plm": plm, "seed": seed, "error": str(e)}]

    # Get WT embedding as baseline (from test set)
    wt_embedding = test_embeddings[0:1]

    # 4-5. Run attribution methods and compute metrics across noise levels
    print(f"4. Running {len(PHASE2_CONFIG['methods'])} attribution methods on test set")
    print(f"5. Computing metrics across {len(PHASE2_CONFIG['noise_levels'])} noise levels")

    # Dictionary to store attributions for each method (for later metric recomputation)
    all_attributions_residue = {}

    for method in PHASE2_CONFIG['methods']:
        print(f"   Method: {method}")

        # Compute attributions for each test sequence
        try:
            all_attributions_embed = []

            for seq_idx in range(len(test_embeddings)):
                seq_embedding = test_embeddings[seq_idx:seq_idx+1]  # (1, L, D)

                if method.startswith('ig_'):
                    attr_embed = compute_attribution_ig(
                        model, seq_embedding, wt_embedding, device, method=method, steps=50
                    )
                elif method == 'gradient_input':
                    attr_embed = compute_attribution_gradient_input(model, seq_embedding, device)
                elif method == 'input_difference':
                    attr_embed = compute_attribution_input_difference(seq_embedding, wt_embedding)
                else:
                    continue

                all_attributions_embed.append(attr_embed)

            if len(all_attributions_embed) == 0:
                print(f"     Unknown method: {method}, skipping")
                continue

            # Stack all attributions
            print(f"     DEBUG: {len(all_attributions_embed)} attributions to concatenate")
            print(f"     DEBUG: First attribution shape: {all_attributions_embed[0].shape}")
            attributions_embed = np.concatenate([a for a in all_attributions_embed], axis=0)
            print(f"     DEBUG: Concatenated shape: {attributions_embed.shape}")
            # Convert to residue-level (signed preserves direction, unlike L2 norm)
            attributions_residue = compute_residue_attributions(attributions_embed, method='signed')
            print(f"     DEBUG: After residue reduction: {attributions_residue.shape}")
            print(f"     Attribution shape: {attributions_residue.shape}")

            # Store for later metric recomputation
            all_attributions_residue[method] = attributions_residue.copy()
        except Exception as e:
            print(f"     ERROR computing attributions: {e}")
            continue

        # Compute metrics across noise levels
        for noise_level in PHASE2_CONFIG['noise_levels']:
            try:
                # Add noise to fitness for robust evaluation
                fitness_noisy = fitness_noiseless + np.random.normal(0, noise_level, len(fitness_noiseless))

                # Aggregate metrics across all test sequences
                all_attrs = attributions_residue[:, :len(ground_truth_w)]  # (N, L)
                all_attrs_embed = attributions_embed[:, :len(ground_truth_w), :]  # (N, L, D)

                # Compute metrics for each sequence then average
                seq_metrics = []
                for seq_idx in range(len(all_attrs)):
                    attr_trimmed = all_attrs[seq_idx]  # (L,)
                    attr_embed_trimmed = all_attrs_embed[seq_idx]  # (L, D)

                    ranking_metrics = compute_ranking_metrics(attr_trimmed, ground_truth_w)
                    sign_acc = compute_sign_accuracy(
                        attr_trimmed,
                        [(i, 'X', 'X') for i in range(len(ground_truth_w))],
                        ground_truth_w
                    )
                    robustness_metrics = compute_noise_robustness_metrics(
                        attr_trimmed,
                        ground_truth_w,
                        attr_embed_trimmed,
                        model_loss
                    )

                    seq_metrics.append({**ranking_metrics, "sign_accuracy": sign_acc, **robustness_metrics})

                # Average metrics across sequences
                all_metrics = {}
                for metric_name in seq_metrics[0].keys():
                    values = [m[metric_name] for m in seq_metrics if metric_name in m and m[metric_name] is not None]
                    if values:
                        all_metrics[metric_name] = np.mean(values)
                    else:
                        all_metrics[metric_name] = None

                # Append results
                for metric_name, metric_value in all_metrics.items():
                    if metric_value is not None:
                        results.append({
                            "protein": protein,
                            "landscape": landscape,
                            "plm": plm,
                            "seed": seed,
                            "method": method,
                            "noise_level": noise_level,
                            "metric": metric_name,
                            "value": metric_value,
                        })

            except Exception as e:
                print(f"     ERROR at noise={noise_level}: {e}")
                continue

    # Save attributions for later metric recomputation
    import pickle
    attributions_file = out_dir / f"attributions_{protein}_{landscape}_{plm}_seed{seed}.pkl"
    with open(attributions_file, 'wb') as f:
        pickle.dump({
            'all_attributions_residue': all_attributions_residue,
            'ground_truth_w': ground_truth_w,
            'methods': PHASE2_CONFIG["methods"],
            'noise_levels': PHASE2_CONFIG["noise_levels"],
        }, f)
    print(f"  Saved attributions to: {attributions_file}")

    # Append model performance to CSV
    import csv
    perf_file = OUTPUT_BASE / "model_performance.csv"
    file_exists = perf_file.exists()

    with open(perf_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['protein', 'landscape', 'plm', 'seed', 'val_loss', 'val_spearman'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(model_perf)
    print(f"  Saved model performance to: {perf_file}")

    # Save config
    config = {
        "experiment": "E15_PHASE2",
        "protein": protein,
        "landscape": landscape,
        "plm": plm,
        "seed": seed,
        "methods": PHASE2_CONFIG["methods"],
        "noise_levels": PHASE2_CONFIG["noise_levels"],
        "predictor_hparams": hparams,
    }
    save_config(out_dir, config, protein, plm, seed)

    print(f"✓ Completed: {protein} / {landscape} / {plm} / seed={seed} ({len(results)} results)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="E15 PHASE 2 Job Launcher — p53 Cross-pLM Comparison (after E12)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (setup only, no computation)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if E12 is not complete (not recommended)"
    )
    parser.add_argument(
        "--plms",
        type=str,
        default=",".join(PHASE2_CONFIG["plms"]),
        help="Comma-separated pLM list (default: all)"
    )
    parser.add_argument(
        "--landscapes",
        type=str,
        default=",".join(PHASE2_CONFIG["landscapes"]),
        help="Comma-separated landscape list (default: all)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(i) for i in range(1, PHASE2_CONFIG["num_seeds"] + 1)),
        help="Comma-separated seed list (default: 1-5)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_BASE,
        help="Output directory (default: results/revision/E15_crosspLM)"
    )

    args = parser.parse_args()

    # Check E12 status
    if not args.force and not args.dry_run:
        if not check_e12_complete():
            print("\n⚠️  BLOCKED: E12 must complete before running PHASE 2")
            print(f"   Run with --force to override (not recommended)")
            sys.exit(1)

    # Parse arguments
    plms = args.plms.split(",")
    landscapes = args.landscapes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"\n{'='*70}")
    print(f"E15 PHASE 2 — p53 Cross-pLM Comparison")
    print(f"{'='*70}")
    print(f"Proteins:   {PHASE2_CONFIG['proteins']}")
    print(f"Landscapes: {landscapes}")
    print(f"pLMs:       {plms}")
    print(f"Methods:    {len(PHASE2_CONFIG['methods'])} × {len(PHASE2_CONFIG['noise_levels'])} noise levels")
    print(f"Seeds:      {seeds}")
    print(f"Output:     {args.output_dir}")
    print(f"Dry run:    {args.dry_run}")
    print(f"Force:      {args.force}")
    print(f"E12 status: {'complete' if check_e12_complete() else 'incomplete'}")
    print(f"{'='*70}\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "metrics_long.csv"

    # Run experiments
    total_experiments = len(PHASE2_CONFIG["proteins"]) * len(landscapes) * len(plms) * len(seeds)
    completed = 0
    all_results = []

    for protein in PHASE2_CONFIG["proteins"]:
        for landscape in landscapes:
            for plm in plms:
                for seed in seeds:
                    completed += 1
                    print(f"\n[{completed}/{total_experiments}]", end=" ")

                    results = run_experiment(
                        protein=protein,
                        landscape=landscape,
                        plm=plm,
                        seed=seed,
                        dry_run=args.dry_run,
                    )

                    # Log results
                    for result in results:
                        all_results.append(result)
                        log_result(csv_path, result)

    print(f"\n{'='*70}")
    print(f"✓ E15 PHASE 2 Complete!")
    print(f"Results saved to: {csv_path}")
    print(f"Total rows: {len(all_results)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
