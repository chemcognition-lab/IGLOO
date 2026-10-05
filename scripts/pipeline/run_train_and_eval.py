#!/usr/bin/env python3
"""
Train-and-Evaluate Runner (no Integrated Gradients)

Identical to run_integrated_gradients.py in every respect — same inputs, same
config format, same output directories, same model saving — but skips the
integrated gradients and visualisation steps entirely.

Usage:
    python scripts/pipeline/run_train_and_eval.py config.yaml
    python scripts/pipeline/run_train_and_eval.py config.yaml training.n_folds=3
    python scripts/pipeline/run_train_and_eval.py config.yaml --validate-only

This is equivalent to running:
    python scripts/pipeline/run_integrated_gradients.py config.yaml --skip-ig
"""

import argparse
import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Reuse everything from the IG runner — no duplication
from run_integrated_gradients import (
    IntegratedGradientsRunner,
    _apply_overrides,
)
from src.igloo.config_utils import load_config, validate_config


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model (no integrated gradients)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/pipeline/run_train_and_eval.py configs/betalactamase_config.yaml
    python scripts/pipeline/run_train_and_eval.py configs/erbc_config.yaml training.n_folds=3
    python scripts/pipeline/run_train_and_eval.py configs/pafa_config.yaml results_dir=results/quick_check
        """,
    )

    parser.add_argument(
        'config_path',
        type=str,
        help='Path to the configuration file (YAML or JSON) — same format as run_integrated_gradients.py',
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the configuration file without running anything',
    )

    parser.add_argument(
        'overrides',
        nargs='*',
        metavar='key.subkey=value',
        help='Optional dot-path overrides applied on top of the config, e.g. '
             'training.n_folds=3  results_dir=results/myjob',
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file '{args.config_path}' not found.")
        sys.exit(1)

    try:
        config = load_config(args.config_path)
        if args.overrides:
            _apply_overrides(config, args.overrides)
        validate_config(config)
        print(f"Configuration file '{args.config_path}' is valid!")

        if args.validate_only:
            print("Validation complete. Exiting.")
            return

        # Run with skip_ig=True — all other behaviour is identical to the IG runner
        runner = IntegratedGradientsRunner(
            args.config_path,
            overrides=args.overrides,
            skip_ig=True,
        )
        results = runner.run_complete_analysis()

        print("\nTrain-and-evaluate completed successfully!")
        return results

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
