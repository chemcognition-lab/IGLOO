#!/usr/bin/env python3
"""
Batch runner for integrated gradients analysis across all datasets and configurations.

This script helps you systematically run all combinations of:
- Datasets: PafA, betalactamase, ERBC
- Methods: embedding, token
- Attribution: magnitude, signed

Usage:
    python run_all_ig_analyses.py [--dataset DATASET] [--method METHOD] [--attribution ATTR]

Examples:
    # Run all configurations for all datasets
    python run_all_ig_analyses.py

    # Run only PafA analyses
    python run_all_ig_analyses.py --dataset pafa

    # Run only token-level analyses
    python run_all_ig_analyses.py --method token

    # Run only signed attribution analyses
    python run_all_ig_analyses.py --attribution signed

    # Run specific combination
    python run_all_ig_analyses.py --dataset betalactamase --method token --attribution signed
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def get_all_configurations():
    """Get all available configuration combinations."""
    datasets = ['pafa', 'betalactamase', 'erbc']
    methods = ['embedding', 'token']
    attributions = ['magnitude', 'signed']

    configs = []
    for dataset in datasets:
        for method in methods:
            for attribution in attributions:
                if dataset == 'pafa':
                    config_name = f"pafa_config_{method}-{attribution}.yaml"
                else:
                    config_name = f"{dataset}_config_{method}-{attribution}.yaml"

                configs.append({
                    'dataset': dataset,
                    'method': method,
                    'attribution': attribution,
                    'config_file': config_name,
                    'full_path': f"configs/{config_name}"
                })

    return configs


def run_analysis(config_path, log_dir):
    """Run a single integrated gradients analysis."""
    config_name = os.path.basename(config_path).replace('.yaml', '')
    log_file = os.path.join(log_dir, f"{config_name}.log")

    print(f"Running analysis: {config_name}")
    print(f"Config: {config_path}")
    print(f"Log file: {log_file}")

    # Create the command
    cmd = [
        sys.executable,
        "scripts/run_integrated_gradients.py",
        config_path
    ]

    try:
        # Run the analysis and capture output
        with open(log_file, 'w') as f:
            f.write(f"Starting analysis at {datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()

            process.wait()

            f.write(f"\nAnalysis completed at {datetime.now()}\n")
            f.write(f"Exit code: {process.returncode}\n")

        if process.returncode == 0:
            print(f"✓ Analysis completed successfully: {config_name}")
            return True
        else:
            print(f"✗ Analysis failed: {config_name} (exit code: {process.returncode})")
            return False

    except Exception as e:
        print(f"✗ Error running analysis {config_name}: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"\nError: {str(e)}\n")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run integrated gradients analysis for all dataset/method combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all configurations
    python run_all_ig_analyses.py

    # Run only PafA analyses
    python run_all_ig_analyses.py --dataset pafa

    # Run only token-level analyses
    python run_all_ig_analyses.py --method token

    # Run only signed attribution
    python run_all_ig_analyses.py --attribution signed

    # Specific combination
    python run_all_ig_analyses.py --dataset betalactamase --method embedding --attribution magnitude
        """
    )

    parser.add_argument(
        '--dataset',
        choices=['pafa', 'betalactamase', 'erbc'],
        help='Run analyses for specific dataset only'
    )

    parser.add_argument(
        '--method',
        choices=['embedding', 'token'],
        help='Run analyses for specific method only'
    )

    parser.add_argument(
        '--attribution',
        choices=['magnitude', 'signed'],
        help='Run analyses for specific attribution method only'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without actually running'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue running other analyses if one fails'
    )

    args = parser.parse_args()

    # Get all configurations
    all_configs = get_all_configurations()

    # Filter based on arguments
    filtered_configs = []
    for config in all_configs:
        if args.dataset and config['dataset'] != args.dataset:
            continue
        if args.method and config['method'] != args.method:
            continue
        if args.attribution and config['attribution'] != args.attribution:
            continue

        # Check if config file exists
        if os.path.exists(config['full_path']):
            filtered_configs.append(config)
        else:
            print(f"Warning: Config file not found: {config['full_path']}")

    if not filtered_configs:
        print("No matching configurations found!")
        sys.exit(1)

    print(f"Found {len(filtered_configs)} configuration(s) to run:")
    for config in filtered_configs:
        print(f"  - {config['config_file']} ({config['dataset']}, {config['method']}, {config['attribution']})")

    if args.dry_run:
        print("\nDry run mode - no analyses will be executed.")
        return

    # Create log directory
    log_dir = "logs/ig_batch_run"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nStarting batch analysis run at {datetime.now()}")
    print(f"Logs will be saved to: {log_dir}")
    print("=" * 60)

    # Run analyses
    results = []
    for i, config in enumerate(filtered_configs, 1):
        print(f"\n[{i}/{len(filtered_configs)}] Starting: {config['config_file']}")

        success = run_analysis(config['full_path'], log_dir)
        results.append({
            'config': config['config_file'],
            'success': success
        })

        if not success and not args.continue_on_error:
            print("\nStopping due to failure. Use --continue-on-error to continue with remaining analyses.")
            break

        print("-" * 60)

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"Total analyses: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed analyses:")
        for result in results:
            if not result['success']:
                print(f"  ✗ {result['config']}")

    if successful > 0:
        print("\nSuccessful analyses:")
        for result in results:
            if result['success']:
                print(f"  ✓ {result['config']}")

    print(f"\nLogs saved in: {log_dir}")
    print(f"Completed at: {datetime.now()}")


if __name__ == "__main__":
    main()
