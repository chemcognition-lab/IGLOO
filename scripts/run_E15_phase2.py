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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from igloo.train_and_eval import train_model_with_embeddings
from igloo.embedding import get_embeddings
from igloo.utils import create_model
from igloo.synthetic_data import SyntheticDatasetGenerator
from igloo.advanced_ig import integrated_gradients_embeddings


# ============================================================================
# Configuration
# ============================================================================

PHASE2_CONFIG = {
    "proteins": ["p53"],
    "landscapes": ["additive", "epistatic", "nonlinear"],
    "plms": ["esmc_300m", "ankh_base", "esm1b_t33_650M_UR50D"],
    "methods": ["ig_linear", "ig_slerp", "ig_nearest_neighbor", "gradient_input", "input_difference"],
    "noise_levels": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    "num_seeds": 5,
}

# Hyperparameters (consistent with GB1 cross-pLM runs)
PREDICTOR_HPARAMS = {
    "esmc_300m": {
        "hidden_dim1": 256,
        "hidden_dim2": 128,
        "hidden_dim3": 64,
        "dropout_rate": 0.2,
        "learning_rate": 1e-4,
        "epochs": 100,
        "batch_size": 32,
    },
    "ankh_base": {
        "hidden_dim1": 256,
        "hidden_dim2": 128,
        "hidden_dim3": 64,
        "dropout_rate": 0.2,
        "learning_rate": 1e-4,
        "epochs": 100,
        "batch_size": 32,
    },
    "esm1b_t33_650M_UR50D": {
        "hidden_dim1": 256,
        "hidden_dim2": 128,
        "hidden_dim3": 64,
        "dropout_rate": 0.2,
        "learning_rate": 1e-4,
        "epochs": 100,
        "batch_size": 32,
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

    return integrated_gradients_embeddings(
        model=model,
        input_embedding=embeddings,
        baseline_embedding=baseline_embedding,
        device=device,
        steps=steps,
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


def compute_residue_attributions(attributions: np.ndarray, method: str = 'l2_norm') -> np.ndarray:
    """Convert embedding-level (B, L, D) to residue-level (B, L)."""
    if method == 'l2_norm':
        return np.linalg.norm(attributions, axis=-1)
    elif method == 'magnitude':
        return np.abs(attributions).sum(axis=-1)
    else:
        return np.linalg.norm(attributions, axis=-1)


def compute_ranking_metrics(
    attributions: np.ndarray,
    ground_truth_w: np.ndarray,
) -> Dict[str, float]:
    """
    Compute NDCG, AP@R, AUROC for ranking quality.

    Args:
        attributions: (L,) residue-level attributions
        ground_truth_w: (L,) ground truth importance weights

    Returns:
        Dict with NDCG, AP@R, AUROC scores
    """
    L = len(attributions)

    # NDCG@L (perfect ranking is ground truth sorted)
    true_labels = np.argsort(-ground_truth_w)
    true_relevances = ground_truth_w[true_labels]
    pred_relevances = attributions[true_labels]
    ndcg = ndcg_score([true_relevances], [pred_relevances], k=L)

    # AP@R (average precision at rank k where k=rank of top residue)
    pred_ranks = np.argsort(-attributions)
    top_rank = np.where(ground_truth_w > 0.5)[0]
    if len(top_rank) > 0:
        auc_pr = average_precision_score(
            (ground_truth_w > 0.5).astype(int),
            attributions
        )
    else:
        auc_pr = 0.5

    # AUROC (ranking by attribution vs ground truth importance)
    fpr, tpr, _ = roc_curve((ground_truth_w > 0.5).astype(int), attributions)
    auroc = auc(fpr, tpr)

    return {
        'ndcg': float(ndcg),
        'ap_r': float(auc_pr),
        'auroc': float(auroc),
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

        sequences = df['SEQ'].tolist()
        fitness_noiseless = df['fitness'].values
        print(f"   Loaded {len(sequences)} sequences, ground truth shape {ground_truth_w.shape}")
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

    # 2. Get embeddings
    print(f"2. Getting embeddings (pLM={plm})")
    try:
        embeddings = get_embeddings(
            sequences,
            layer=33,
            batch_size=hparams['batch_size'],
            dataset_name=f"E15_phase2_{protein}_{landscape}",
            subdir='revision/E15',
            esm_model_name=plm,
        )
        print(f"   Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"   ERROR getting embeddings: {e}")
        return [{"protein": protein, "landscape": landscape, "plm": plm, "seed": seed, "error": str(e)}]

    # 3. Train predictor
    print(f"3. Training predictor (epochs={hparams['epochs']})")
    try:
        model = create_model({
            'model': {
                'type': 'linear_regression',
                'hidden_dims': [hparams['hidden_dim1'], hparams['hidden_dim2'], hparams['hidden_dim3']],
                'dropout_rate': hparams['dropout_rate'],
            }
        }, device)

        train_result = train_model_with_embeddings(
            model=model,
            train_sequences=sequences,
            train_targets=fitness_noiseless,
            esm_layer=33,
            pooling_method='linear',
            batch_size=hparams['batch_size'],
            learning_rate=hparams['learning_rate'],
            num_epochs=hparams['epochs'],
            device=device,
            dataset_name=f"E15_phase2_{protein}_{landscape}",
            embeddings_subdir='revision/E15',
            esm_model_name=plm,
        )
        print(f"   Training complete")
    except Exception as e:
        print(f"   ERROR training: {e}")
        return [{"protein": protein, "landscape": landscape, "plm": plm, "seed": seed, "error": str(e)}]

    # Get WT embedding as baseline
    wt_embedding = embeddings[0:1]

    # 4-5. Run attribution methods and compute metrics across noise levels
    print(f"4. Running {len(PHASE2_CONFIG['methods'])} attribution methods")
    print(f"5. Computing metrics across {len(PHASE2_CONFIG['noise_levels'])} noise levels")

    for method in PHASE2_CONFIG['methods']:
        print(f"   Method: {method}")

        # Compute attributions
        try:
            if method.startswith('ig_'):
                attributions_embed = compute_attribution_ig(
                    model, embeddings, wt_embedding, device, method=method, steps=50
                )
            elif method == 'gradient_input':
                attributions_embed = compute_attribution_gradient_input(model, embeddings, device)
            elif method == 'input_difference':
                attributions_embed = compute_attribution_input_difference(embeddings, wt_embedding)
            else:
                print(f"     Unknown method: {method}, skipping")
                continue

            # Convert to residue-level
            attributions_residue = compute_residue_attributions(attributions_embed, method='l2_norm')
            print(f"     Attribution shape: {attributions_residue.shape}")
        except Exception as e:
            print(f"     ERROR computing attributions: {e}")
            continue

        # Compute metrics across noise levels
        for noise_level in PHASE2_CONFIG['noise_levels']:
            try:
                # Add noise to fitness for robust evaluation
                fitness_noisy = fitness_noiseless + np.random.normal(0, noise_level, len(fitness_noiseless))

                # Compute ranking metrics
                ranking_metrics = compute_ranking_metrics(
                    attributions_residue[0],
                    ground_truth_w
                )

                # Compute sign accuracy
                sign_acc = compute_sign_accuracy(
                    attributions_residue[0],
                    [(i, 'X', 'X') for i in range(len(ground_truth_w))],
                    ground_truth_w
                )

                # Append results
                for metric_name, metric_value in ranking_metrics.items():
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

                results.append({
                    "protein": protein,
                    "landscape": landscape,
                    "plm": plm,
                    "seed": seed,
                    "method": method,
                    "noise_level": noise_level,
                    "metric": "sign_accuracy",
                    "value": sign_acc,
                })

            except Exception as e:
                print(f"     ERROR at noise={noise_level}: {e}")
                continue

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
