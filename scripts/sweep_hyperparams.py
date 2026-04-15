#!/usr/bin/env python3
"""
W&B Hyperparameter Sweep for IGLOO

Runs a Bayesian (or grid/random) hyperparameter sweep over the model training
parameters defined in a sweep config YAML.  The IG analysis is intentionally
*skipped* so each trial only pays the cost of data loading + K-fold CV.

Usage
-----
# 1. Initialise a new sweep and immediately start one agent process:
    python scripts/sweep_hyperparams.py \\
        configs/sweep_config.yaml \\
        configs/synthetic/T4Lysozyme_additive_config.yaml

# 2. Start additional agents for an existing sweep (e.g. on multiple GPUs):
    python scripts/sweep_hyperparams.py \\
        configs/sweep_config.yaml \\
        configs/synthetic/T4Lysozyme_additive_config.yaml \\
        --sweep-id <SWEEP_ID>

# 3. Limit the number of runs per agent:
    python scripts/sweep_hyperparams.py \\
        configs/sweep_config.yaml \\
        configs/synthetic/T4Lysozyme_additive_config.yaml \\
        --count 20
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.config_utils import load_config, validate_config
from src.igloo.data_processing import load_and_prepare_data, prepare_wildtype_sequence, split_dataset
from src.igloo.train_and_eval import kfold_cv_and_train
from src.igloo.utils import evaluate_model_performance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(relative: str) -> str:
    """Return absolute path relative to the project root."""
    return str(_PROJECT_ROOT / relative)


def _build_config(base_config: dict, sweep_params: dict) -> dict:
    """Merge W&B sweep parameters into a copy of the base IGLOO config.

    All keys in *sweep_params* that match ``model.*`` are written into
    ``config['model']``; keys matching ``training.*`` go into
    ``config['training']``.  Unrecognised top-level keys are ignored.
    """
    cfg = copy.deepcopy(base_config)

    model_keys    = {
        'hidden_dim1', 'hidden_dim2', 'hidden_dim3', 'hidden_dim4',
        'dropout_rate', 'learning_rate', 'batch_size', 'epochs',
        'loss_type', 'pooling_method', 'patience', 'min_epochs',
    }
    training_keys = {'n_folds', 'test_fraction', 'split_seed'}

    for key, value in sweep_params.items():
        if key in model_keys:
            cfg['model'][key] = value
        elif key in training_keys:
            cfg.setdefault('training', {})[key] = value
        # else: silently skip unknown keys

    # Always force retrain when sweeping
    cfg['force_retrain'] = True

    return cfg


# ---------------------------------------------------------------------------
# Core training function called by each W&B agent run
# ---------------------------------------------------------------------------

def _train(base_config: dict, device: torch.device) -> None:
    """Single sweep trial: load data → CV → log metrics.  No IG analysis."""

    with wandb.init() as run:
        sweep_params = dict(wandb.config)
        cfg = _build_config(base_config, sweep_params)

        dataset_name = cfg['data']['name']
        run.name = (
            f"{dataset_name}"
            f"_lr{cfg['model']['learning_rate']:.0e}"
            f"_d{cfg['model']['dropout_rate']}"
            f"_h{cfg['model']['hidden_dim1']}"
        )

        # ── Log resolved hyperparameters ──────────────────────────────────
        wandb.config.update({
            'dataset':        dataset_name,
            'pooling_method': cfg['model']['pooling_method'],
            'loss_type':      cfg['model']['loss_type'],
        }, allow_val_change=True)

        # ── Data loading ──────────────────────────────────────────────────
        data_path = _resolve_path(cfg['data']['path'])
        (df, sequences, y_raw, y,
         y_log_mean, y_log_std) = load_and_prepare_data(cfg, data_path)

        # ── Train / val split ────────────────────────────────────────────
        training_cfg  = cfg.get('training', {})
        test_fraction = float(training_cfg.get('test_fraction', 0.0))
        split_seed    = int(training_cfg.get('split_seed', 42))

        if test_fraction > 0:
            trainval_idx, test_idx = split_dataset(
                df, test_fraction=test_fraction, seed=split_seed
            )
            tv_sequences = [sequences[i] for i in trainval_idx]
            tv_y         = y[trainval_idx]
        else:
            tv_sequences = sequences
            tv_y         = y
            test_idx     = np.array([], dtype=int)

        # ── K-fold CV ─────────────────────────────────────────────────────
        n_folds = int(training_cfg.get('n_folds', 5))

        # Temporary results dir (not used for saving during sweep)
        sweep_results_dir = _resolve_path(
            f"results/sweep/{dataset_name}/{run.id}"
        )
        os.makedirs(sweep_results_dir, exist_ok=True)

        embeddings_subdir = (
            'synthetic' if 'synthetic' in cfg['data'].get('path', '') else None
        )

        final_model, cv_summary, model_cfg, _norm = kfold_cv_and_train(
            config=cfg,
            sequences=tv_sequences,
            y=tv_y,
            results_dir=sweep_results_dir,
            device=device,
            n_folds=n_folds,
            save_path=None,          # don't persist sweep models by default
            embeddings_subdir=embeddings_subdir,
        )

        # ── Log CV summary metrics ────────────────────────────────────────
        log_dict: dict = {}
        for k, v in cv_summary.items():
            if k == 'per_fold':
                continue
            if isinstance(v, (int, float)):
                log_dict[k] = v

        # ── Optional: evaluate on held-out test set ──────────────────────
        if len(test_idx) > 0:
            test_sequences = [sequences[i] for i in test_idx]
            test_y         = y[test_idx]
            test_y_raw     = y_raw[test_idx]

            test_r2, test_pearson, test_spearman, _ = evaluate_model_performance(
                final_model, test_sequences, test_y, test_y_raw,
                y_log_mean, y_log_std, cfg,
                sweep_results_dir, device,
                split_name='test',
            )
            log_dict['test_r2']          = test_r2
            log_dict['test_pearson_r']   = test_pearson
            log_dict['test_spearman_rho'] = test_spearman

        wandb.log(log_dict)

        # Primary metric (used by Bayesian optimiser)
        primary = log_dict.get('mean_val_r2', log_dict.get('test_r2'))
        if primary is not None:
            wandb.summary['best_metric'] = primary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a W&B hyperparameter sweep for IGLOO model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        'sweep_config',
        help='Path to the W&B sweep config YAML (defines parameter search space).',
    )
    p.add_argument(
        'base_config',
        help='Path to the IGLOO dataset config YAML (same format as run_integrated_gradients.py).',
    )
    p.add_argument(
        '--sweep-id',
        default=None,
        metavar='ID',
        help='Attach to an existing sweep instead of creating a new one.',
    )
    p.add_argument(
        '--project',
        default=None,
        metavar='NAME',
        help='W&B project name (overrides value in sweep config if set).',
    )
    p.add_argument(
        '--entity',
        default=None,
        metavar='NAME',
        help='W&B entity (team/user) name.',
    )
    p.add_argument(
        '--count',
        type=int,
        default=None,
        metavar='N',
        help='Maximum number of runs for this agent (None = run until sweep finishes).',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load configs ──────────────────────────────────────────────────────
    sweep_config = load_config(args.sweep_config)
    base_config  = load_config(args.base_config)
    validate_config(base_config)

    # Allow CLI overrides of project / entity
    if args.project:
        sweep_config['project'] = args.project
    if args.entity:
        sweep_config['entity'] = args.entity

    # Default project name to dataset name if not set
    sweep_config.setdefault('project', f"igloo-{base_config['data']['name']}")

    # ── Initialise or attach to sweep ─────────────────────────────────────
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Attaching to existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            sweep_config,
            project=sweep_config.get('project'),
            entity=sweep_config.get('entity'),
        )
        print(f"Created sweep: {sweep_id}")

    # ── Launch agent ──────────────────────────────────────────────────────
    wandb.agent(
        sweep_id,
        function=lambda: _train(base_config, device),
        project=sweep_config.get('project'),
        entity=sweep_config.get('entity'),
        count=args.count,
    )


if __name__ == '__main__':
    main()
