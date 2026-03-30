#!/usr/bin/env python3
"""
Unified Integrated Gradients Analysis Runner

A configuration-driven script for running integrated gradients analysis
on protein sequence data using ESM embeddings and various ML models.

Usage:
    python run_integrated_gradients.py config.yaml
    python run_integrated_gradients.py config.json

Example:
    python run_integrated_gradients.py configs/betalactamase_config.yaml
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# Import analysis modules
from src.igloo.config_utils import (
    load_config,
    select_sequences_for_analysis,
    validate_config,
)
from src.igloo.data_processing import load_and_prepare_data, prepare_wildtype_sequence
from src.igloo.integrated_gradients import run_integrated_gradients_analysis
from src.igloo.train_and_eval import train_or_load_model
from src.igloo.utils import (
    create_visualizations_and_exports,
    evaluate_model_performance,
    print_completion_summary,
)


class IntegratedGradientsRunner:
    """Unified runner for integrated gradients analysis with config support."""

    def __init__(self, config_path: str):
        """Initialize the runner with a configuration file."""
        self.config = load_config(config_path)
        validate_config(self.config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize attributes that will be set during execution
        self.df = None
        self.sequences = None
        self.y = None
        self.y_raw = None
        self.y_log_mean = None
        self.y_log_std = None
        self.wt_seq = None
        self.final_model = None
        self.model_cfg = None

    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        # Construct data path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        data_path = os.path.join(project_root, self.config['data']['path'])

        # load_and_prepare_data returns a tuple: (df, sequences, y_raw, y, y_log_mean, y_log_std)
        result = load_and_prepare_data(self.config, data_path)
        self.df, self.sequences, self.y_raw, self.y, self.y_log_mean, self.y_log_std = result

    def prepare_wildtype_sequence(self):
        """Prepare the wildtype sequence for analysis."""
        self.wt_seq, self.wt_idx = prepare_wildtype_sequence(self.df, self.config)

    def train_or_load_model(self):
        """Train a new model or load an existing one."""
        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        results_dir = os.path.join(
            project_root,
            self.config.get('results_dir', f"results/{self.config['data']['name']}_ig")
        )
        os.makedirs(results_dir, exist_ok=True)

        print(f"Results directory: {results_dir}")

        # train_or_load_model returns a tuple: (final_model, train_model, train_metrics)
        self.final_model, train_model, train_metrics = train_or_load_model(
            self.config, self.sequences, self.y, results_dir, self.device
        )
        self.model_cfg = self.config['model']  # Use the model config from the configuration

        # Print model status
        if train_model:
            print("✓ Model training completed successfully")
            if 'train_r2' in train_metrics:
                print(f"  Training R² = {train_metrics['train_r2']:.4f}")
        else:
            print("✓ Pre-trained model loaded successfully")

    def evaluate_model_performance(self):
        """Evaluate the trained model's performance."""
        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        return evaluate_model_performance(
            self.final_model, self.sequences, self.y, self.y_raw,
            self.y_log_mean, self.y_log_std, self.config, results_dir, self.device
        )

    def run_integrated_gradients_analysis(self, eval_results=None):
        """Run the integrated gradients analysis."""
        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        # Get sequences to analyze based on configuration
        selection_config = self.config.get('sequence_selection', {
            'include_extremes': True,  # Include lowest and highest activity sequences
            'include_median': True,    # Include median activity sequence
            'include_top_n': 0,        # Number of additional top sequences (beyond highest)
            'include_bottom_n': 0,     # Number of additional bottom sequences (beyond lowest)
            'custom_sequences': [],    # Custom sequences to include
            'max_error_percent': 10.0  # Maximum prediction error for top_n/bottom_n filtering
        })

        # Extract predictions from eval_results if available for error filtering
        y_pred = None
        if eval_results is not None:
            # eval_results is a tuple: (r2, pearson_r, spearman_rho, y_pred_denorm)
            y_pred = eval_results[3]  # Get the denormalized predictions

        sequences_to_analyze = select_sequences_for_analysis(
            self.df, self.y_raw, self.wt_seq, selection_config,
            model=self.final_model, sequences=self.sequences, y_pred=y_pred
        )

        print(f"\nSelected {len(sequences_to_analyze)} sequences for integrated gradients analysis:")
        for name, seq, _baseline in sequences_to_analyze:
            # Find the activity value for this sequence
            try:
                seq_idx = self.sequences.index(seq)
                activity = self.y_raw[seq_idx]
                print(f"  {name}: activity = {activity:.4f}")
            except ValueError:
                print(f"  {name}: custom sequence (not in training data)")

        return run_integrated_gradients_analysis(
            sequences_to_analyze, self.config, self.y_raw, self.sequences,
            self.final_model, results_dir, self.device, self.y_log_mean, self.y_log_std
        )

    def create_visualizations_and_exports(self, results, eval_results):
        """Create visualizations and export results."""
        # Extract required metrics from eval_results tuple
        r2, pearson_r, spearman_rho, y_pred_denorm = eval_results

        # Create basic train metrics dict
        train_metrics = {'note': 'metrics_from_evaluation'}

        # Construct results directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        default_results_dir = f"results/{self.config['data']['name']}_ig"
        results_dir = os.path.join(project_root, self.config.get('results_dir', default_results_dir))

        create_visualizations_and_exports(
            results, r2, pearson_r, spearman_rho, train_metrics,
            self.config, results_dir, self.wt_seq, self.final_model
        )

    def _write_timing_report(self, timing: dict, results_dir: str) -> None:
        """Write a timing summary to a .txt file in the results directory."""
        os.makedirs(results_dir, exist_ok=True)
        report_path = os.path.join(results_dir, f"{self.config['data']['name']}_timing_report.txt")

        total = timing.get('total', 0.0)
        n_ig = timing.get('n_ig_sequences', 0)

        lines = [
            "=" * 60,
            "PIPELINE TIMING REPORT",
            "=" * 60,
            f"Dataset      : {self.config['data']['name']}",
            f"Started at   : {timing.get('start_time', 'N/A')}",
            f"Finished at  : {timing.get('end_time', 'N/A')}",
            f"Device       : {self.device}",
            "-" * 60,
            f"{'Step':<40} {'Duration':>10}",
            "-" * 60,
        ]

        step_keys = [
            ('data_loading',        'Data loading & preparation'),
            ('wildtype_prep',       'Wildtype sequence preparation'),
            ('embedding_retrieval', 'Embedding retrieval (if needed)'),
            ('model_train_load',    'Model training / loading'),
            ('model_evaluation',    'Model evaluation'),
            ('integrated_gradients', f'Integrated gradients ({n_ig} sequence(s))'),
            ('visualizations',      'Visualizations & exports'),
        ]

        for key, label in step_keys:
            secs = timing.get(key)
            if secs is not None:
                lines.append(f"  {label:<38} {self._fmt_duration(secs):>10}")

        lines += [
            "-" * 60,
            f"  {'TOTAL':<38} {self._fmt_duration(total):>10}",
            "=" * 60,
        ]

        report = "\n".join(lines) + "\n"
        print(report)

        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Timing report saved to {report_path}")

    @staticmethod
    def _fmt_duration(secs: float) -> str:
        """Format seconds as a human-readable string."""
        if secs < 60:
            return f"{secs:.1f}s"
        m, s = divmod(int(secs), 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def run_complete_analysis(self):
        """Run the complete integrated gradients analysis pipeline."""
        timing: dict = {}
        pipeline_start = time.perf_counter()
        timing['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            print("="*60)
            print("INTEGRATED GRADIENTS ANALYSIS PIPELINE")
            print("="*60)
            print(f"Config file: {self.config.get('analysis', {}).get('dataset_name', 'Unknown')}")
            print(f"Device: {self.device}")
            print()

            # Step 1: Load and prepare data
            _t = time.perf_counter()
            self.load_and_prepare_data()
            timing['data_loading'] = time.perf_counter() - _t

            # Step 2: Prepare wildtype sequence
            _t = time.perf_counter()
            self.prepare_wildtype_sequence()
            timing['wildtype_prep'] = time.perf_counter() - _t

            # Step 3: Train or load model
            _t = time.perf_counter()
            self.train_or_load_model()
            timing['model_train_load'] = time.perf_counter() - _t
            timing['embedding_retrieval'] = 0.0

            # Step 4: Evaluate model performance
            _t = time.perf_counter()
            eval_results = self.evaluate_model_performance()
            timing['model_evaluation'] = time.perf_counter() - _t

            # Step 5: Run integrated gradients analysis
            _t = time.perf_counter()
            ig_results = self.run_integrated_gradients_analysis(eval_results)
            timing['integrated_gradients'] = time.perf_counter() - _t
            # Count analysed sequences (exclude internal keys starting with '_')
            timing['n_ig_sequences'] = sum(
                1 for k in (ig_results or {}) if not k.startswith('_')
            )

            # Step 6: Create visualizations and exports
            _t = time.perf_counter()
            if ig_results:
                self.create_visualizations_and_exports(ig_results, eval_results)
            timing['visualizations'] = time.perf_counter() - _t

            timing['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timing['total'] = time.perf_counter() - pipeline_start

            # Step 7: Print completion summary
            r2, pearson_r, spearman_rho, _ = eval_results
            eval_results_dict = {
                'r2': r2,
                'pearson_r': pearson_r,
                'spearman_rho': spearman_rho
            }

            print_completion_summary(
                len(self.sequences),
                len(ig_results) if ig_results else 0,
                eval_results_dict,
                self.config.get('results_dir', 'results'),
                self.config['data']['name']
            )

            # Step 8: Write timing report
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
            results_dir = os.path.join(
                project_root,
                self.config.get('results_dir', f"results/{self.config['data']['name']}_ig")
            )
            self._write_timing_report(timing, results_dir)

            return {
                'evaluation': eval_results,
                'integrated_gradients': ig_results,
                'timing': timing,
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run integrated gradients analysis with configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_integrated_gradients.py configs/betalactamase_config.yaml
    python run_integrated_gradients.py configs/erbc_config.json
    python run_integrated_gradients.py configs/pafa_config.yaml
        """
    )

    parser.add_argument(
        'config_path',
        type=str,
        help='Path to the configuration file (YAML or JSON)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the configuration file without running analysis'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file '{args.config_path}' not found.")
        sys.exit(1)

    try:
        # Load and validate config
        config = load_config(args.config_path)
        validate_config(config)
        print(f"Configuration file '{args.config_path}' is valid!")

        if args.validate_only:
            print("Validation complete. Exiting.")
            return

        # Run analysis
        runner = IntegratedGradientsRunner(args.config_path)
        results = runner.run_complete_analysis()

        print("\nAnalysis completed successfully!")
        return results

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
